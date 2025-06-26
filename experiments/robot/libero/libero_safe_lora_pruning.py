import torch
import pdb
from torch import nn
from transformers import AutoModelForCausalLM

from transformers import AutoModelForCausalLM
from experiments.robot.robot_utils import get_model

# Setup dummy config with checkpoint
class DummyConfig():
    def __init__(self, pretrained_checkpoint):
        self.model_family = "openvla"
        self.pretrained_checkpoint = pretrained_checkpoint
        self.load_in_8bit = False
        self.load_in_4bit = False
        self.pruned_inference = False
        self.load_to_cpu = True

def nudge_and_save(pruned_model, dense_model, save_dir='pruned_model_nudged', tau=0.8, device='cuda'):
    # Move models to appropriate devices
    pruned_model.eval()
    dense_sd = dense_model.state_dict()

    # Perform patch under no_grad to avoid in-place grad errors
    with torch.no_grad():
        for name, module in pruned_model.named_modules():
            if isinstance(module, nn.Linear):
                print(f"Checking Safety Gap for {name}")
                Wp = module.weight 
                Wd = dense_sd[name + ".weight"]

                # Compute the gap
                V = Wd - Wp

                # SVD: get top singular component
                U, S, Vh = torch.linalg.svd(V)
                r = 25  # number of components to keep, must be << d

                # Slice off the top-r singular triplets
                U_r   = U[:, :r]        # (d_out × r)
                S_r   = S[:r]           # (r,)
                Vh_r  = Vh[:r, :]       # (r × d_in)

                # Sum up each rank-1 piece
                delta_W = sum(
                    S_r[i] * torch.ger(U_r[:, i], Vh_r[i, :])
                    for i in range(r)
                )

                # Rank-1 patch
                # delta_W = sigma1 * torch.ger(u1, v1)

                # Update weight in-place on its .data buffer
                module.weight.data.add_(delta_W)
    
    # Save the patched model
    pruned_model.save_pretrained(save_dir)
    print(f"Patched model saved to {save_dir}")

# Load the pruned OpenVLA model
print("[*] Loading pruned OpenVLA model...")
path_to_pruned_model = "/workspace/models/openvla-7b-pruned-2_4-disabled-sparse-compression"
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


