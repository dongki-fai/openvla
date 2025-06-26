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

# def nudge_weights(pruned_model, dense_model, device='cpu'):
#     for (fqn_pruned, module_pruned), (fqn_dense, module_dense) in zip(pruned_model.named_modules(), dense_model.named_modules()):
#         if isinstance(module_pruned, nn.Linear): 
#             if module_pruned.weight.shape[0] % 32 == 0 and module_pruned.weight.shape[1] % 64 == 0:
#                 print(f"Checking Safety Gap for {fqn_pruned}")

#                 # Get the dense and pruned weights
#                 W_dense = module_dense.weight.detach()
#                 W_pruned = module_pruned.weight.detach()

#                 # Compute Alignment Matrix
#                 alignment_matrix = (W_dense - W_pruned)

#                 U, S, Vh = torch.linalg.svd(alignment_matrix)
#                 u1 = U[:,   0]         # (d_out,)
#                 v1 = Vh[0, :]          # (d_in,)
#                 sigma1 = S[0].item()   # scalar

#                 # low-rank patch
#                 delta_W = sigma1 * torch.ger(u1, v1)  # rank-1 approximation

#                 print(f'Delta_W: {delta_W[:6, :6]})')
#                 # Nudge the original pruned model's weight
#                 module_pruned.weight.add_(delta_W)
#                 print(f"Nudged: {module_pruned[:6, :6]})")

#     pruned_model.save_pretrained('SparseGPT_nudged') 



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
                u1 = U[:, 0]            # (out_dim,)
                v1 = Vh[0, :]           # (in_dim,)
                sigma1 = S[0]

                # Rank-1 patch
                delta_W = sigma1 * torch.ger(u1, v1)

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


