import torch
import pandas as pd
import pdb
import json
import csv
from torch import nn
from transformers import AutoModelForCausalLM

from transformers import AutoModelForCausalLM
from experiments.robot.robot_utils import get_model


RANK = 500  # Number of components to keep for nudging
# IGNORE_SPECIFIC_LANGUAGE_LAYERS = True
# LANGUAGE_LAYERS_TO_IGNORE = list(range(0, 16))
CHOOSE_SINGULAR_VALUES_BY = 'Magnitude' # 'Magnitude' or 'Random'
TOTALLY_REPLACE_PRUNED_WEIGHTS = False  # Whether to replace pruned weights with nudged weights or add them
SAVE_SINGULAR_VALUES = False 
SAVE_RANDOM_INDICES = False

# Setup dummy config with checkpoint
class DummyConfig():
    def __init__(self, pretrained_checkpoint):
        self.model_family = "openvla"
        self.pretrained_checkpoint = pretrained_checkpoint
        self.load_in_8bit = False
        self.load_in_4bit = False
        self.pruned_inference = False
        self.load_to_cpu = True


if SAVE_SINGULAR_VALUES:
    # CSV setup
    csv_path = "singular_values.csv"
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["layer","singular_values"])
        writer.writeheader()
elif SAVE_RANDOM_INDICES:
    # CSV setup for random indices
    csv_path = "singular_values_chosen_indices.csv"
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["layer", "chosen_indices"])
        writer.writeheader()

def nudge_and_save(pruned_model, dense_model, save_dir='pruned_model_nudged', tau=0.8, device='cuda'):
    # Move models to appropriate devices
    pruned_model.eval()
    dense_sd = dense_model.state_dict()

    device_gpu = torch.device(device)
    device_cpu = torch.device('cpu')

    # Perform patch under no_grad to avoid in-place grad errors
    with torch.no_grad():
        for name, module in pruned_model.named_modules():
            if isinstance(module, nn.Linear):

                # if IGNORE_SPECIFIC_LANGUAGE_LAYERS: 
                #     if "language_model" in name:
                #         layer_id = int(name.split("language_model.model.layers.")[1].split(".")[0])
                #         if layer_id in LANGUAGE_LAYERS_TO_IGNORE:
                #             continue
                #         else:
                #             pass 
                # else:
                #     pass

                # Move ONLY this submodule to GPU  
                module.to(device_gpu)

                Wp = module.weight
                Wd = dense_sd[name + ".weight"].to(device_gpu)

                if not TOTALLY_REPLACE_PRUNED_WEIGHTS:
                    print(f"Checking Safety Gap for {name}")
                    # Compute the gap
                    V = Wd - Wp
                else:
                    V = Wd

                # SVD: get top singular component
                U, S, Vh = torch.linalg.svd(V)
                print("SVD singular values:", S)
                r = RANK  # number of components to keep, must be << d

                if CHOOSE_SINGULAR_VALUES_BY == 'Random':
                    n_vals = S.size(0)
                    # pick r random, non-repeating indices
                    rand_idx = torch.randperm(n_vals, device=S.device)[:r]

                    # extract those components
                    U_r  = U[:, rand_idx]        # (d_out × r)
                    S_r  = S[rand_idx]           # (r,)
                    Vh_r = Vh[rand_idx, :]       # (r × d_in)

                elif CHOOSE_SINGULAR_VALUES_BY == 'Magnitude':
                    # Slice off the top-r singular triplets
                    U_r   = U[:, :r]        # (d_out × r)
                    S_r   = S[:r]           # (r,)
                    Vh_r  = Vh[:r, :]       # (r × d_in)
                else:
                    raise ValueError("CHOOSE_SINGULAR_VALUES_BY must be 'Magnitude' or 'Random'")

                # Sum up each rank-1 piece
                delta_W = sum(
                    S_r[i] * torch.ger(U_r[:, i], Vh_r[i, :])
                    for i in range(r)
                )

                # print(f"The SVD", delta_W[:8, :8])  # Print the top-left 8x8 block of delta_W
                # print(f"The Wp", Wp[:8, :8])  # Print the top-left 8x8 block of Wp

                # Rank-1 patch
                # delta_W = sigma1 * torch.ger(u1, v1)
                
                if not TOTALLY_REPLACE_PRUNED_WEIGHTS:
                    print(f" -> Adding nudged weights to pruned weights for {name}")
                    # Update weight in-place on its .data buffer
                    # module.weight.data.add_(delta_W)

                    mask = (Wp != 0).to(Wp.dtype)            # 1s where Wp is nonzero, 0s elsewhere
                    delta_masked = delta_W * mask            # zero out all entries outside the original support

                    # print(f"Mask", delta_masked[:8, :8])  # Print the top-left 8x8 block of delta_masked
                    module.weight.data.add_(delta_masked)    # now W_new = Wp + Δ only on the mask


                else:
                    print(f" -> Replacing pruned weights with SVD weights for {name}")
                    module.weight.data = delta_W


                # Move this submodule back to CPU
                module.to(device_cpu)

                if SAVE_SINGULAR_VALUES:
                    # append to CSV
                    with open(csv_path, "a", newline="") as csvfile:
                        writer = csv.DictWriter(csvfile, fieldnames=["layer","singular_values"])
                        writer.writerow({
                            "layer": name,
                            "singular_values": json.dumps(S.tolist())
                        })
                elif SAVE_RANDOM_INDICES:
                    # save the random indices we picked
                    with open(csv_path, "a", newline="") as csvfile:
                        writer = csv.DictWriter(csvfile, fieldnames=["layer", "chosen_indices"])
                        writer.writerow({
                            "layer": name,
                            "chosen_indices": json.dumps(rand_idx.tolist())
                        })
                
    # Save the patched model
    pruned_model.save_pretrained(save_dir)
    print(f"Patched model saved to {save_dir}")

# Load the pruned OpenVLA model
print("[*] Loading pruned OpenVLA model...")
path_to_pruned_model = "/workspace/models/openvla-7b-pruned-2_4-Wanda-pruned-language_backbone-calibset-5TotalWindowGripper"
cfg = DummyConfig(path_to_pruned_model)
pruned_model = get_model(cfg)
pruned_model = pruned_model.float()

print("[*] Loading dense OpenVLA model...")
path_to_dense_model = "/workspace/models/openvla-7b-finetuned-libero-spatial"
# path_to_dense_model = "/workspace/openvla/pruned_model_nudged"
cfg = DummyConfig(path_to_dense_model)
dense_model = get_model(cfg)
dense_model = dense_model.float()

nudge_and_save(pruned_model, dense_model, device='cuda')

pdb.set_trace()


# --- Method According to Formulation ---
# print(f"Shape of V: {V.shape}")
# print(f"Rank of V: {torch.linalg.matrix_rank(V)}")
# # Compute the projector C
# C = V @ V.t()    
# normV = V.norm()
# C = C / normV
# print(f"Shape of C: {C.shape}")
# print(f"Rank of C: {torch.linalg.matrix_rank(C)}")
# --- C is actually not a rank 1 matrix, but very high rank ---


