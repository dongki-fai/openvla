from experiments.robot.robot_utils import get_model
from experiments.robot.openvla_utils import get_openvla_processor

from llmcompressor.modifiers.pruning import WandaPruningModifier, MagnitudePruningModifier
from llmcompressor.modifiers.obcq import SparseGPTModifier
from llmcompressor import oneshot

import torch

from transformers import AutoModelForCausalLM
from transformers.data import default_data_collator
from experiments.robot.robot_utils import get_model
from transformers import AutoProcessor
from experiments.robot.openvla_utils import get_openvla_processor

import torch
from PIL import Image
import io
import os
import random
import tensorflow as tf
from transformers import AutoModelForCausalLM
import pdb


if not hasattr(torch, "OutOfMemoryError"):
    class _OOM(RuntimeError): pass
    torch.OutOfMemoryError = _OOM

from datasets import IterableDataset
from datasets import Dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier

PRUNING_MODIFIER = "Wanda"  # ["Wanda", "Magnitude", or "SparseGPT"]

# Only one of these should be True at a time
PRUNE_VISION_BACKBONE = False
PRUNE_LANGUAGE_MODEL = True
PRUNE_FULL_MODEL = False

IGNORE_SPECIFIC_LANGUAGE_LAYERS = False 
# Half are: list(range(0, 16)) or list(range(16, 32))
LANGUAGE_LAYERS_TO_IGNORE = list(range(16, 32))
NUM_CALIB_SAMPLES = 15000

assert sum([PRUNE_VISION_BACKBONE, PRUNE_LANGUAGE_MODEL, PRUNE_FULL_MODEL]) == 1, \
    "Only one of PRUNE_* flags can be True at a time."

# Determine what parts to prune
if PRUNE_FULL_MODEL:
    ignore = ["re:^language_model\\.lm_head\\."] # or None
elif PRUNE_VISION_BACKBONE:
    ignore = [
        "re:^language_model\\.",
        "re:^projector\\.",
        "re:^language_model\\.lm_head\\."
    ]
elif PRUNE_LANGUAGE_MODEL:
    ignore = [
        "re:^vision_backbone\\.",
        "re:^projector\\.",
        "re:^language_model\\.lm_head\\."
    ]
else:
    raise ValueError("No pruning target selected!")

if IGNORE_SPECIFIC_LANGUAGE_LAYERS:
    ignore += [f"re:^language_model\\.model\\.layers\\.{i}\\." for i in LANGUAGE_LAYERS_TO_IGNORE]

if not hasattr(torch, "OutOfMemoryError"):
    class _OOM(RuntimeError): pass
    torch.OutOfMemoryError = _OOM

class DummyConfig():
    def __init__(self, pretrained_checkpoint="/workspace/models/openvla-7b-finetuned-libero-spatial"):
        self.model_family = "openvla"
        self.pretrained_checkpoint = pretrained_checkpoint
        self.load_in_8bit = False
        self.load_in_4bit = False
        self.pruned_inference = False
        self.load_to_cpu = False

def build_model_inputs(device, processor, image, instruction):
    """
    Returns the kwargs dict you would normally pass directly to
    `OpenVLAForActionPrediction.forward(...)`.
    """
    inputs = processor(images=image, text=instruction, return_tensors="pt")
    pixel_values   = inputs["pixel_values"].to(device, dtype=torch.bfloat16)
    input_ids      = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    return {
        "pixel_values":   pixel_values.squeeze(0),   # (3,224,224)  bf16
        "input_ids":      input_ids.squeeze(0),      # (T,)
        "attention_mask": attention_mask.squeeze(0)  # (T,)
    }

def decode_jpeg(jpeg_bytes):
    """Decode JPEG bytes into RGB image as np.uint8 array."""
    image = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
    return image

def build_calibration_dataset_from_examples(
    tfrecord_paths,
    device,
    processor,
    num_samples,
    seed=42,
):
    calibration_set = []

    rng = random.Random(seed)
    paths = list(tfrecord_paths)
    rng.shuffle(paths)

    for tfrecord_path in paths:
        print(f"[*] Scanning: {tfrecord_path}")
        dataset = tf.data.TFRecordDataset(tfrecord_path)

        for record in dataset:
            if len(calibration_set) >= num_samples:
                break

            try:
                # use SequenceExample and loop over all steps
                seq = tf.train.SequenceExample()
                seq.ParseFromString(record.numpy())
                ctx = seq.context.feature

                instruction = ctx["steps/language_instruction"].bytes_list.value[0].decode()
                img_bytes_list = ctx["steps/observation/image"].bytes_list.value  # list of all frames

                for img_bytes in img_bytes_list:
                    if len(calibration_set) >= num_samples:
                        break
                    image = decode_jpeg(img_bytes)
                    calibration_set.append(build_model_inputs(device, processor, image, instruction))

            except Exception as e:
                print(f"[!] Skipping record due to error: {e}")
                continue

        if len(calibration_set) >= num_samples:
            break

    print(f"[✓] Collected {len(calibration_set)} calibration examples.")
    return calibration_set


def calibration_generator(
        tfrecord_paths,
        processor,
        num_samples,
        device,
        seed=42,
    ):

    rng = random.Random(seed)
    paths = list(tfrecord_paths)
    rng.shuffle(paths)
    count = 0

    for tfrecord_path in paths:
        print(f"[*] Scanning: {tfrecord_path}")
        dataset = tf.data.TFRecordDataset(tfrecord_path)
        for record in dataset:
            if count >= num_samples:
                return

            try:
                # use SequenceExample and loop over all steps
                seq = tf.train.SequenceExample()
                seq.ParseFromString(record.numpy())
                ctx = seq.context.feature

                instruction = ctx["steps/language_instruction"].bytes_list.value[0].decode()
                img_bytes_list = ctx["steps/observation/image"].bytes_list.value  # list of all frames

                for img_bytes in img_bytes_list:
                    if count >= num_samples:
                        return
                    image = decode_jpeg(img_bytes)
                    yield build_model_inputs(device, processor, image, instruction)
                    count += 1

            except Exception as e:
                print(f"[!] Skipping record due to error: {e}")
                continue


if __name__ == "__main__":

    cfg = DummyConfig()

    # Load the full OpenVLA model
    print("[*] Loading full OpenVLA model...")
    model = get_model(cfg)

    model = model.float()  # ensure model is in float32 for calibration

    # Get the processor
    processor = get_openvla_processor(cfg)

    # tfrecord_dir = "/workspace/data/modified_libero_rlds/libero_spatial_no_noops/1.0.0"
    # tfrecord_dir = "/workspace/data/closed_gripper_libero_rlds/libero_spatial_no_noops/1.0.0"
    tfrecord_dir = "/workspace/data/closed_gripper_2_5_window_libero_rlds/libero_spatial_no_noops/1.0.0"

    tfrecord_paths = [
        os.path.join(tfrecord_dir, f)
        for f in sorted(os.listdir(tfrecord_dir))
        if ".tfrecord" in f
    ]

    calib_data = build_calibration_dataset_from_examples(
        tfrecord_paths=tfrecord_paths,
        device=model.device,
        processor=processor,
        num_samples=NUM_CALIB_SAMPLES,
    )

    print("Number of Calibration Samples", len(calib_data))

    ds = Dataset.from_list(calib_data).with_format("torch")
    del calib_data
    print(len(ds), ds.column_names)        # sanity-check

    # # build a streaming HF IterableDataset (no giant Python list)
    # gen = lambda: calibration_generator(tfrecord_paths, processor, NUM_CALIB_SAMPLES, device=model.device)
    # ds  = IterableDataset.from_generator(gen).with_format("torch")
    # print("[✓] Streaming dataset ready")
    
    # Create the pruner
    if PRUNING_MODIFIER == "Wanda":
        pruner = WandaPruningModifier(
            targets="Linear",
            sparsity=0.5,
            mask_structure="2:4",
            ignore=ignore,
        )
    elif PRUNING_MODIFIER == "Magnitude":
        pruner = MagnitudePruningModifier(
            targets="Linear",
            init_sparsity=0.0,
            final_sparsity=0.5,
            mask_structure="unstructured",  # Does not support "2:4" mask structure
            ignore=ignore,
        )
    elif PRUNING_MODIFIER == "SparseGPT":
        pruner = SparseGPTModifier(
            targets="Linear",
            sparsity=0.5,
            mask_structure="2:4", 
            ignore=ignore,
        )
    else:
        raise ValueError(f"Unknown PRUNING_MODIFIER: {PRUNING_MODIFIER}")

    oneshot(model=model,
            recipe=pruner,
            dataset=ds,
            num_calibration_samples=NUM_CALIB_SAMPLES,
            pipeline='basic',
            save_compressed=False,
            output_dir="delete_me",)

    if PRUNE_FULL_MODEL:
        pruned_scope = "full_model"
    elif PRUNE_VISION_BACKBONE:
        pruned_scope = "vision_backbone"
    elif PRUNE_LANGUAGE_MODEL:
        pruned_scope = "language_backbone"
    else:
        raise ValueError("No pruning target selected!")


    save_dir = f"openvla-7b-pruned-2_4-{PRUNING_MODIFIER}-pruned-{pruned_scope}-Closed_Gripper_Data"

    if IGNORE_SPECIFIC_LANGUAGE_LAYERS:
        save_dir += f"-ignore-lang-layers-{min(LANGUAGE_LAYERS_TO_IGNORE)}-{max(LANGUAGE_LAYERS_TO_IGNORE)}"
    
    total_params = 0
    zero_params  = 0
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            W = module.weight.data
            total_params += W.numel()
            zero_params  += (W == 0).sum().item()
    print(f"Global zero fraction: {zero_params/total_params:.3%}")


    # save in compressed form
    model.save_pretrained(
        save_dir,
        safe_serialization=True,          # safetensors
        # save_compressed=True,             # write bit-masks instead of dense tensors
        # compression_scheme="sparse-24-bitmask",   # this string matters!
        disable_sparse_compression=True,
    )

    processor.save_pretrained(save_dir)
    print(f"[✓] Model and processor saved to {save_dir}")


    pdb.set_trace()

# ###############################################

# ## Sanity Check Visualization of Pruned Weights

# ################################################
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.colors import ListedColormap, BoundaryNorm

# # pick a layer, e.g. the first Linear in your language model
# # layer = model.language_model.model.layers[0].self_attn.q_proj 
# # layer = model.vision_backbone.featurizer.blocks[0].mlp.fc1
# layer = model.vision_backbone.featurizer.blocks[0].attn.qkv

# W = layer.weight.data.cpu().numpy()   # shape (out_dim, in_dim)
## W = layer.weight.data.to(torch.float32).cpu().numpy()

# # make a binary mask: 0 where W==0, 1 everywhere else
# binary = (W != 0).astype(int)

# # define a 2-color map: zeros in light gray, non-zeros in navy
# cmap = ListedColormap(["#FFFFFF", "#F15A29"])
# norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)  # boundaries at -0.5→0.5→1.5

# plt.figure(figsize=(8, 6))
# plt.imshow(binary, aspect="auto", cmap=cmap, norm=norm)
# plt.colorbar(ticks=[0, 1],label="Zero vs Non-Zero",format=plt.FuncFormatter(lambda val, loc: "zero" if val == 0 else "non-zero"))
# plt.title("Binary mask of Vision Backbone attn.qkv weight (layer 0)")
# plt.xlabel("Input dimension")
# plt.ylabel("Output dimension")

# # save high-res copy before showing
# plt.savefig("vision_backbone_attn_qkv.png", dpi=300, bbox_inches="tight")


# # --------- Zoomed In View of Matrix

# # slice the first 100×100
# W_small = W[:20, :20]

# # make a binary mask: 0 where W_small==0, 1 where !=0
# binary_small = (W_small != 0).astype(int)

# plt.figure(figsize=(6, 6))
# plt.imshow(binary_small, aspect="equal", cmap=cmap, norm=norm)
# plt.colorbar(ticks=[0, 1],label="Zero vs Non-Zero",format=plt.FuncFormatter(lambda val, loc: "zero" if val == 0 else "non-zero"))
# plt.title("Binary mask of Vision Backbone attn.qkv weight")
# plt.xlabel("Input dim (0–19)")
# plt.ylabel("Output dim (0–19)")

# # save hi-res
# plt.savefig("vision_backbone_attn_qkv_20x20.png", dpi=300, bbox_inches="tight")
