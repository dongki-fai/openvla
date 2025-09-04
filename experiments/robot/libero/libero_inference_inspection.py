import os
import math
import time
import tensorflow as tf
import random
from PIL import Image
import io
import torch
from torch import nn
import torch.nn.functional as F
from collections import defaultdict
from datasets import Dataset
from tqdm.auto import tqdm
import pdb

from experiments.robot.robot_utils import get_model, get_processor, import_neccessary_libraries
from experiments.robot.openvla_utils import get_openvla_processor
from experiments.robot.pruning_utils import attach_sparse_kernel, wrap_linears_with_svd, compile_linears

from torch.sparse import to_sparse_semi_structured, SparseSemiStructuredTensor
import gc

SparseSemiStructuredTensor._FORCE_CUTLASS = True

FILTER_FOR = 'language_model'  # Filter for specific layers to apply SVD and apply sparse kernel
SKIP_LAYERS = ['vision_backbone', 'lm_head', 'projector']

# Setup dummy config with checkpoint
class DummyConfig():
    def __init__(self, pretrained_checkpoint):
        self.model_family = "openvla"
        self.pretrained_checkpoint = pretrained_checkpoint
        self.load_in_8bit = False
        self.load_in_4bit = False
        self.pruned_inference = False
        self.load_to_cpu = False


def input_generator(tfrecord_paths, seed=42):
    rng = random.Random(seed)
    paths = list(tfrecord_paths)
    rng.shuffle(paths)

    for tfrecord_path in paths:
        print(f"[*] Scanning: {tfrecord_path}")
        dataset = tf.data.TFRecordDataset(tfrecord_path)

        for record in dataset:
            try:
                # use SequenceExample and loop over all steps
                seq = tf.train.SequenceExample()
                seq.ParseFromString(record.numpy())
                ctx = seq.context.feature

                instruction = ctx["steps/language_instruction"].bytes_list.value[0].decode()
                img_bytes_list = ctx["steps/observation/image"].bytes_list.value  # list of all frames

                for img_bytes in img_bytes_list:
                    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

                    # yield last image (same behavior as your original loop)
                    yield image, instruction

            except Exception as e:
                print(f"[!] Skipping record due to error: {e}")
                continue


# define a function to register forward hooks on all Linear layers
def register_linear_input_hooks(model):
    # layer_name -> list of captured input blocks (each with shape [d_in, B_block])
    buffers = defaultdict(list)
    # store hook handles so we can remove them later
    handles = []

    # iterate through all submodules (with hierarchical names)
    for name, mod in model.named_modules():
        # only attach hooks to Linear layers
        if isinstance(mod, torch.nn.Linear):
            if any(skip_layer in name for skip_layer in SKIP_LAYERS): 
                continue

            def make_hook(layer_name):
                # define the hook called every time this layer runs forward
                def hook(model, input, output):
                    # take the input tensor to this Linear and detach from autograd graph
                    x = input[0].detach()

                    # if "model.layers.18.self_attn.q_proj" in layer_name:
                    #     print(f"Captured input for layer {layer_name}:")
                    #     print("Input Shape:", input[0].shape)
                    # print("Output Shape:", output.shape)
                    # --------- skip prefill ----------
                    if x.shape[1] > 1:         # S > 1 ⇒ prefill
                        return

                    buffers[layer_name].append(x.cpu())
                # return the actual hook function
                return hook
            
            # attach the hook to this Linear layer
            h = mod.register_forward_hook(make_hook(name))
            # remember the handle so we can remove the hook later
            handles.append(h)

    # return both the handles and the per-layer buffers of captured inputs
    return handles, buffers

def run_model(model, tfrecord_paths, processor, num_samples, seed=42):
    inference_count = 0

    model.to("cuda")

    handles, buffers = register_linear_input_hooks(model)

    pbar = tqdm(total=num_samples, desc="Inference", unit="sample")

    avg_inference_time = 0.0
    for image, instruction in input_generator(tfrecord_paths, seed=seed):

        if inference_count >= num_samples:
            break
        
        inputs = processor(images=image, text=instruction, return_tensors="pt")
        inputs.to("cuda").to(torch.float16)
        start_time = time.time()
        action = model.predict_action(**inputs, use_cache=False)
        end_time = time.time()
        print(f"Action: {action}")
        print(f"Inference Time: {end_time - start_time:.4f} seconds")
        avg_inference_time += (end_time - start_time)
        inference_count += 1
        pbar.update(1)

    pbar.close()

    # # remove hooks and collate per layer
    # for h in handles: 
    #     h.remove()

    print(f"Average Inference Time: {avg_inference_time / num_samples:.4f} seconds")

    # del buffers, handles
    # torch.cuda.empty_cache()


if __name__ == "__main__":

    print("[*] Loading dense OpenVLA model...")
    path_to_model = "/workspace/models/openvla-7b-finetuned-libero-spatial"
    # path_to_model = "/workspace/models/openvla-7b-pruned-2_4-Wanda-pruned-language_backbone-calibset-5TotalWindowGripper"
    cfg = DummyConfig(path_to_model)
    cfg.use_cache = False
    import_neccessary_libraries(cfg.model_family)

    model = get_model(cfg)
    # Convert model to float16 for inference
    model = model.to(torch.float16)

    # Get the processor
    processor = get_processor(cfg)

    tfrecord_dir = "/workspace/data/closed_gripper_2_5_window_libero_rlds/libero_spatial_no_noops/1.0.0"
    tfrecord_paths = [
        os.path.join(tfrecord_dir, f)
        for f in sorted(os.listdir(tfrecord_dir))
        if ".tfrecord" in f
    ]

    # # Attach sparse kernel
    # model = attach_sparse_kernel(model, filter_for=FILTER_FOR, skip_layers=SKIP_LAYERS)

    # # Load SVD factors and wrap linears
    # svd_factors_path = "/workspace/models/svd_factors_libero_spatial_lb_5total/svd_factors_rank_200.pt"
    # model = wrap_linears_with_svd(model, svd_factors_path, filter_for=FILTER_FOR, skip_layers=SKIP_LAYERS, dtype=torch.float16, device="cuda")

    # Compile the Model
    # model = torch.compile(model, mode="max-autotune")
    model = compile_linears(model)

    run_model(
        model=model,
        tfrecord_paths=tfrecord_paths,
        processor=processor,
        num_samples=100,
    )

