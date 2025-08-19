import os
import math
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

from experiments.robot.robot_utils import get_model
from experiments.robot.openvla_utils import get_processor

NUM_CALIB_SAMPLES = 4000

SKIP_LAYERS = ['vision_backbone', 'lm_head', 'projector']

def get_target_linear_layer_names(model):
    names = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            if any(skip_layer in name for skip_layer in SKIP_LAYERS):
                continue
            names.append(name)
    return names

# Setup dummy config with checkpoint
class DummyConfig():
    def __init__(self, pretrained_checkpoint):
        self.model_family = "openvla"
        self.pretrained_checkpoint = pretrained_checkpoint
        self.load_in_8bit = False
        self.load_in_4bit = False
        self.pruned_inference = False
        self.load_to_cpu = True


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

                    # print(f"Captured input for layer {layer_name}:")
                    # print("Input Shape:", input[0].shape)
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

def collect_inputs(model, tfrecord_paths, processor, num_samples, seed=42):
    inference_count = 0

    model.to("cuda")

    handles, buffers = register_linear_input_hooks(model)

    pbar = tqdm(total=num_samples, desc="Calib inference", unit="sample")

    for image, instruction in input_generator(tfrecord_paths, seed=seed):

        if inference_count >= num_samples:
            break

        inputs = processor(images=image, text=instruction, return_tensors="pt")
        inputs.to("cuda")
        action = model.predict_action(**inputs)

        # print(f"Action: {action}")

        inference_count += 1
        pbar.update(1)

    model.cpu()  # move model back to CPU after inference

    print(f"[✓] Collected Inputs from {inference_count} calibration examples.")
    pbar.close()

    # remove hooks and collate per layer
    for h in handles: 
        h.remove()

    # Gives you a dict of layer_name -> [Number Tokens, d_in]
    X_map = {layer_name: (torch.cat(layer_input, dim=1).squeeze() if len(layer_input) > 0 else None)
                for layer_name, layer_input in buffers.items()}
    
    # Move to trash
    del buffers, handles
    torch.cuda.empty_cache()

    return X_map


def masked_grad_update_layer(
    Wp: torch.Tensor,   # pruned weight [d_out, d_in]
    Wd: torch.Tensor,   # dense teacher [d_out, d_in]
    X:  torch.Tensor,   # inputs [d_in, B_total]
    steps: int = 100,
    lr: float = 1e-2,
    ) -> torch.Tensor:
    """
    Gradient-based masked fit of W to match Wd on inputs X.
    Preserves the exact 2:4 zeros by masking grads + weights every step.
    """
    device = Wp.device
    mask =  (Wp != 0).to(device).float()
    Wp = Wp.detach()
    X  = X.detach().to(device).float()
    Wd = Wd.detach().to(device).float()

    # trainable copy (keeps optimizer numerics stable), initialized from Wp
    Wp_new = torch.nn.Parameter(Wp.detach().to(device).float().clone())

    optimizer = torch.optim.Adam([Wp_new], lr=lr)

    for s in range(steps):
        optimizer.zero_grad(set_to_none=True)

        # teacher & current outputs for this chunk
        y_d = X @ Wd.T # X [N Tokens, d_in], Wd.T [d_in, d_out] -> y_d [N Tokens, d_out]
        y_p = X @ Wp_new.T # X [N Tokens, d_in], Wp_new.T [d_in, d_out] -> y_p [N Tokens, d_out]
        loss = F.mse_loss(y_p, y_d)

        if s == 0: print(f"Start Loss: {loss.item()} at step {s+1}/{steps}")

        # full-batch
        loss.backward()

        # keep grads only on the surviving weights
        if Wp_new.grad is not None:
            Wp_new.grad.mul_(mask)

        optimizer.step()

        # re-enforce exact zeros so the mask never changes
        with torch.no_grad():
            Wp_new.data.mul_(mask)

    # Move to trash 
    del optimizer, Wd, X
    torch.cuda.empty_cache()

    print(f"Final Loss: {loss.item()} after {steps} steps.")
    return Wp_new.data.to(Wp.dtype)

def apply_masked_ls(pruned_model: nn.Module,
                    dense_model: nn.Module,
                    X_map: dict,
                    device: str = "cuda",
                    limit_layers: int | None = 3, 
                    layers_per_batch: int = 30):
    """
    Args:
      pruned_model: the 2:4 model we’re correcting (weights updated in-place)
      dense_model : the teacher (we only read its weights)
      X_map       : dict[layer_name] -> X matrix [d_in, B_total] from hooks
      device      : "cuda" or "cpu"
      lam         : ridge strength (use small value like 1e-3; 0.0 = plain LS)
      limit_layers: update only the first N Linear layers (for quick sanity checks)
      layers_per_batch: number of layers to update in each batch (for memory)

    Returns:
      count of layers updated
    """
    pruned_model.eval()
    pruned_model.to(device)
    dense_sd = dense_model.state_dict()
    updated = 0

    target_layers = get_target_linear_layer_names(pruned_model)
    num_batches = math.ceil(len(target_layers) / layers_per_batch)
    print(f"[*] Will process {len(target_layers)} Linear layers in {num_batches} batch(es) of {layers_per_batch}.")


    for name, module in pruned_model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if any(skip_layer in name for skip_layer in SKIP_LAYERS): 
            print("[!] Skipping layer:", name)
            continue
        if name not in X_map or X_map[name] is None:
            continue

        # fetch weights and captured inputs on device
        module.to(device)
        Wp = module.weight
        Wd = dense_sd[name + ".weight"].to(device)
        X  = X_map[name].to(device)

        # apply masked least squares update
        W_new = masked_grad_update_layer(Wp, Wd, X)

        module.weight.data = W_new

        updated += 1
        print(f"[✓] Updated {name}: W {tuple(Wp.shape)}, X {tuple(X.shape)}")

        if limit_layers is not None and updated >= limit_layers:
            break

    print(f"[✓] Masked LS applied to {updated} layer(s).")

    # Save the patched model
    pruned_model.save_pretrained('teacher_adjusted_model')
    print(f"Patched model saved to teacher_adjusted_model")
    pdb.set_trace()

if __name__ == "__main__":

    print("[*] Loading dense OpenVLA model...")
    path_to_dense_model = "/workspace/models/openvla-7b-finetuned-libero-spatial"
    # path_to_dense_model = "/workspace/openvla/pruned_model_nudged"
    cfg = DummyConfig(path_to_dense_model)
    dense_model = get_model(cfg)
    dense_model = dense_model.float()

    # Get the processor
    processor = get_processor(cfg)

    tfrecord_dir = "/workspace/data/closed_gripper_2_5_window_libero_rlds/libero_spatial_no_noops/1.0.0"
    tfrecord_paths = [
        os.path.join(tfrecord_dir, f)
        for f in sorted(os.listdir(tfrecord_dir))
        if ".tfrecord" in f
    ]

    collected_inputs = collect_inputs(
        model=dense_model,
        tfrecord_paths=tfrecord_paths,
        processor=processor,
        num_samples=NUM_CALIB_SAMPLES,
    )

    # Load the pruned OpenVLA model
    print("[*] Loading pruned OpenVLA model...")
    path_to_pruned_model = "/workspace/models/openvla-7b-pruned-2_4-Wanda-pruned-language_backbone-calibset-5TotalWindowGripper"
    cfg = DummyConfig(path_to_pruned_model)
    pruned_model = get_model(cfg)
    pruned_model = pruned_model.float()

    apply_masked_ls(
        pruned_model=pruned_model,
        dense_model=dense_model,
        X_map=collected_inputs,
        device='cuda',
        limit_layers=None,  # set to None to update all layers
        layers_per_batch=30,   # tune for memory
    )

    pdb.set_trace()