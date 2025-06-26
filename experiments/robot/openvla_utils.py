"""Utils for evaluating the OpenVLA policy."""

import json
import os
import time

import numpy as np
import tensorflow as tf
import torch
from PIL import Image
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

from torch.sparse import to_sparse_semi_structured, SparseSemiStructuredTensor
import gc
from torch import nn

SparseSemiStructuredTensor._FORCE_CUTLASS = True

# Initialize important constants and pretty-printing mode in NumPy.
ACTION_DIM = 7
DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

# Initialize system prompt for OpenVLA v0.1.
OPENVLA_V01_SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)

def patched_from_dense(cls, original_tensor: torch.Tensor):
    cls._validate_device_dim_dtype_shape(original_tensor)
    (
        sparse_tensor_cutlass,
        meta_tensor_cutlass,
    ) = sparse_semi_structured_from_dense_cutlass(original_tensor)

    shape = original_tensor.shape
    requires_grad = original_tensor.requires_grad
    # Free original tensor as soon as it's no longer needed
    del original_tensor
    torch.cuda.empty_cache()

    return cls(
        shape=shape,
        packed=sparse_tensor_cutlass,
        meta=meta_tensor_cutlass,
        packed_t=None,
        meta_t=None,
        compressed_swizzled_bitmask=None,
        requires_grad=False,
    )

SparseSemiStructuredTensor.from_dense = classmethod(patched_from_dense)

def get_vla(cfg):
    """Loads and returns a VLA model from checkpoint."""
    # Load VLA checkpoint.
    print("[*] Instantiating Pretrained VLA model")


    # Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    attn_implementation = None

    # if cfg.pruned_inference:
    #     attn_implementation = None
    # else:
    #     attn_implementation = "flash_attention_2"
    #     print("[*] Loading in BF16 with Flash-Attention Enabled")

    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.pretrained_checkpoint,
        attn_implementation=attn_implementation,
        torch_dtype=torch.bfloat16,
        load_in_8bit=cfg.load_in_8bit,
        load_in_4bit=cfg.load_in_4bit,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )


    # Move model to device.
    # Note: `.to()` is not supported for 8-bit or 4-bit bitsandbytes models, but the model will
    #       already be set to the right devices and casted to the correct dtype upon loading.
    if not cfg.load_in_8bit and not cfg.load_in_4bit and not cfg.load_to_cpu:
        vla = vla.to(DEVICE)

    # TODO: Implement this cleanly
    # PRUNE_BACKBONE_ONLY = True
    # prune_backbone_name = 'language_model' # or 'language_model' or 'vision_backbone'
    # raise NotImplementedError("Integrate pruning only backbone functionality here cleanly.")

    # if cfg.pruned_inference:
    #     for fqn, module in vla.named_modules():
    #         if isinstance(module, nn.Linear): #and "layer" in fqn:
    #             if prune_backbone_name in fqn:
    #                 if module.weight.shape[0] % 32 == 0 and module.weight.shape[1] % 64 == 0:
    #                     print(f"Converting {fqn} to sparse semi-structured")
    #                     # module.weight = nn.Parameter(to_sparse_semi_structured(module.weight))
    #                     old_weight = module.weight.detach()
    #                     sparse_weight = to_sparse_semi_structured(old_weight)
    #                     module.weight = nn.Parameter(sparse_weight)
    #                     del old_weight, sparse_weight
    #                     torch.cuda.empty_cache()
    #                     gc.collect()
    #                 else:
    #                     print(f"[Dimension Error] Skipping {fqn} as it does not have the right dimensions for sparse semi-structured conversion.")
    #             else:
    #                 print(f"[Not Language Backbone] Skipping {fqn} as it is not a linear layer in the language backbone.")
    #         else:
    #             print(f"Skipping {fqn} as it is not a linear layer in the transformer blocks.")


    # Load dataset stats used during finetuning (for action un-normalization).
    dataset_statistics_path = os.path.join(cfg.pretrained_checkpoint, "dataset_statistics.json")
    if os.path.isfile(dataset_statistics_path):
        with open(dataset_statistics_path, "r") as f:
            norm_stats = json.load(f)
        vla.norm_stats = norm_stats
    else:
        print(
            "WARNING: No local dataset_statistics.json file found for current checkpoint.\n"
            "You can ignore this if you are loading the base VLA (i.e. not fine-tuned) checkpoint."
            "Otherwise, you may run into errors when trying to call `predict_action()` due to an absent `unnorm_key`."
        )

    return vla


def get_processor(cfg):
    """Get VLA model's Hugging Face processor."""
    processor = AutoProcessor.from_pretrained(cfg.pretrained_checkpoint, trust_remote_code=True)
    return processor


def crop_and_resize(image, crop_scale, batch_size):
    """
    Center-crops an image to have area `crop_scale` * (original image area), and then resizes back
    to original size. We use the same logic seen in the `dlimp` RLDS datasets wrapper to avoid
    distribution shift at test time.

    Args:
        image: TF Tensor of shape (batch_size, H, W, C) or (H, W, C) and datatype tf.float32 with
               values between [0,1].
        crop_scale: The area of the center crop with respect to the original image.
        batch_size: Batch size.
    """
    # Convert from 3D Tensor (H, W, C) to 4D Tensor (batch_size, H, W, C)
    assert image.shape.ndims == 3 or image.shape.ndims == 4
    expanded_dims = False
    if image.shape.ndims == 3:
        image = tf.expand_dims(image, axis=0)
        expanded_dims = True

    # Get height and width of crop
    new_heights = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))
    new_widths = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))

    # Get bounding box representing crop
    height_offsets = (1 - new_heights) / 2
    width_offsets = (1 - new_widths) / 2
    bounding_boxes = tf.stack(
        [
            height_offsets,
            width_offsets,
            height_offsets + new_heights,
            width_offsets + new_widths,
        ],
        axis=1,
    )

    # Crop and then resize back up
    image = tf.image.crop_and_resize(image, bounding_boxes, tf.range(batch_size), (224, 224))

    # Convert back to 3D Tensor (H, W, C)
    if expanded_dims:
        image = image[0]

    return image


def get_vla_action(vla, processor, base_vla_name, obs, task_label, unnorm_key, center_crop=False):
    """Generates an action with the VLA policy."""
    image = Image.fromarray(obs["full_image"])
    image = image.convert("RGB")

    # (If trained with image augmentations) Center crop image and then resize back up to original size.
    # IMPORTANT: Let's say crop scale == 0.9. To get the new height and width (post-crop), multiply
    #            the original height and width by sqrt(0.9) -- not 0.9!
    if center_crop:
        batch_size = 1
        crop_scale = 0.9

        # Convert to TF Tensor and record original data type (should be tf.uint8)
        image = tf.convert_to_tensor(np.array(image))
        orig_dtype = image.dtype

        # Convert to data type tf.float32 and values between [0,1]
        image = tf.image.convert_image_dtype(image, tf.float32)

        # Crop and then resize back to original size
        image = crop_and_resize(image, crop_scale, batch_size)

        # Convert back to original data type
        image = tf.clip_by_value(image, 0, 1)
        image = tf.image.convert_image_dtype(image, orig_dtype, saturate=True)

        # Convert back to PIL Image
        image = Image.fromarray(image.numpy())
        image = image.convert("RGB")

    # Build VLA prompt
    if "openvla-v01" in base_vla_name:  # OpenVLA v0.1
        prompt = (
            f"{OPENVLA_V01_SYSTEM_PROMPT} USER: What action should the robot take to {task_label.lower()}? ASSISTANT:"
        )
    else:  # OpenVLA
        prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:"

    # Process inputs.
    inputs = processor(prompt, image).to(DEVICE, dtype=torch.bfloat16)

    # normal vla model
    action = vla.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False, use_cache=True)

    return action
