import os
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import pdb
from datasets import Dataset

from transformers import AutoModelForCausalLM
from experiments.robot.robot_utils import get_model
from experiments.robot.openvla_utils import get_openvla_processor
from experiments.robot.libero.libero_pruning import build_calibration_dataset_from_examples
from experiments.robot.libero.libero_pruning import DummyConfig

RANK = 200  # Number of components to keep for nudging
EPOCHS = 20  # Number of epochs for imitation training
LEARNING_RATE = 1e-3  # Learning rate for imitation training
NUM_DATA_SAMPLES = 100  # Number of calibration samples to use
save_with_custom_nudged_layers = False # Set to True to save NudgedLinear layers instead of plain Linear

# class NudgedLinear(nn.Module):
#     def __init__(self, 
#                 pruned_linear: nn.Linear,
#                 U_r: torch.Tensor,
#                 S_r: torch.Tensor, 
#                 Vh_r: torch.Tensor):
#         """
#         U_r:  (d_out, r)
#         S_r:  (r,)
#         Vh_r: (r, d_in)
#         """
#         super().__init__()
#         # keep the original pruned weight frozen
#         self.register_buffer("Wp", pruned_linear.weight.data.clone())
#         # store the frozen bases
#         self.register_buffer("U_r", U_r)
#         self.register_buffer("Vh_r", Vh_r)
#         # make the top-r singular values trainable
#         self.sigma = nn.Parameter(S_r.clone())
#         # copy other module attributes (bias etc)
#         self.bias = pruned_linear.bias
# def forward(self, x):
#     # sparse pruned pass on the *original* weight
#     y1 = F.linear(x, self.Wp, self.bias)

#     # dense low-rank correction
#     delta_W = (self.U_r * self.sigma.unsqueeze(0)) @ self.Vh_r
#     y2 = F.linear(x, delta_W, None)

#     return y1 + y2


# class NudgedLinear(nn.Module):
#     def __init__(self,
#                  orig: nn.Linear,
#                  U_r: torch.Tensor,
#                  S_r: torch.Tensor,
#                  Vh_r: torch.Tensor):
#         super().__init__()
#         # Freeze the original layer in place:
#         orig.weight.requires_grad = False
#         if orig.bias is not None:
#             orig.bias.requires_grad = False

#         # Alias the original weight & bias (no clone):
#         self.register_buffer("W_p", orig.weight.data)
#         self.bias = orig.bias

#         # Store only the small rank-r factors:
#         self.register_buffer("U_r", U_r)           # (d_out, r)
#         self.register_buffer("Vh_r", Vh_r)         # (r, d_in)
#         self.sigma = nn.Parameter(S_r)             # (r,)

class NudgedLinear(nn.Module):
    def __init__(self,
                orig: nn.Linear,
                U_r: torch.Tensor,
                S_r: torch.Tensor, 
                Vh_r: torch.Tensor):
        super().__init__()
        # freeze the original
        orig.weight.requires_grad = False
        if orig.bias is not None:
            orig.bias.requires_grad = False

        # alias it (no copy)
        self.W_p  = orig.weight   # shape (d_out, d_in)
        self.bias = orig.bias

        # store only small factors
        self.register_buffer("U_r",  U_r)    # (d_out,  r)
        self.register_buffer("Vh_r", Vh_r)   # (r,      d_in)
        self.sigma   = nn.Parameter(S_r)     # (r,)

    def forward(self, x):
        # pruned pass
        y1 = F.linear(x, self.W_p, self.bias)  

        # low-rank correction
        z  = x.matmul(self.Vh_r.T)            
        z  = z * self.sigma                  
        y2 = z.matmul(self.U_r.T)             
        return y1 + y2



# def forward(self, x):
#     # x: [B, N, D_in]
#     B, N, D_in = x.shape

#     # 1) Sparse pruned pass as a true sparse-matrix multiply
#     # Flatten into a 2-D matrix of shape (B*N) × D_in,
#     # then transpose to D_in × (B*N) so we can do (out×in) × (in×BN).
#     x_flat    = x.reshape(-1, D_in)                    # [B*N, D_in]
#     x_flat_T  = x_flat.transpose(0, 1)                 # [D_in, B*N]
#     y1_flat_T = torch.sparse.mm(self.Wp, x_flat_T)     # [D_out, B*N]
#     y1_flat   = y1_flat_T.transpose(0, 1)              # [B*N, D_out]
#     y1        = y1_flat.reshape(B, N, -1)             # [B, N, D_out]

#     # 2) Low-rank dense correction pass (any leading dims OK)
#     delta_W = (self.U_r * self.sigma.unsqueeze(0)) @ self.Vh_r  # [D_out, D_in]
#     y2       = F.linear(x, delta_W, None)                      # [B, N, D_out]

#     return y1 + y2


def build_nudged_model(pruned_model, dense_model, rank=100, device="cuda"):
    """
    Copies pruned_model, and for each nn.Linear, computes
    its U_r,S_r,Vh_r from (dense-pruned) and wraps it in NudgedLinear.
    Returns both the nudged_model and a copy of the dense_model for 
    reference in the imitation loss.
    """
    pruned_model.eval()
    dense_sd = dense_model.state_dict()

    test_count = 0

    with torch.no_grad():
        for name, module in pruned_model.named_modules():

            if isinstance(module, nn.Linear):

                # if test_count > 5: 
                #     break
                # else:
                #     test_count += 1

                print(f"Checking Safety Gap for {name}")
                Wp = module.weight
                Wd = dense_sd[name + ".weight"]

                # Compute the gap
                V = (Wd.to(device) - Wp.to(device))

                # compute truncated SVD on V
                U, S, Vh = torch.linalg.svd(V)
                r = rank # number of components to keep, must be << d
                
                # Slice off the top-r singular triplets
                U_r   = U[:, :r].cpu()        # (d_out × r)
                S_r   = S[:r].cpu()           # (r,)
                Vh_r  = Vh[:r, :].cpu()       # (r × d_in)


                # clean up GPU memory immediately
                del V, U, S, Vh
                torch.cuda.empty_cache()


                # replace with our nudged version
                parent, attr = name.rsplit(".", 1)
                setattr(
                    dict(pruned_model.named_modules())[parent],
                    attr,
                    NudgedLinear(module, U_r, S_r, Vh_r)
                )

                # Clean up CPU memory
                del Wp, Wd, U_r, S_r, Vh_r

    return pruned_model


def export_to_dense_linear(nudged_model: nn.Module):
    """
    Replace every NudgedLinear in-place with a standard nn.Linear
    whose weight = W_p + (U_r * sigma) @ Vh_r, and the same bias.
    """
    # map from full module name → module instance
    name2mod = dict(nudged_model.named_modules())

    for name, module in list(nudged_model.named_modules()):
        if isinstance(module, NudgedLinear):
            # 1) build the low-rank delta
            #    (U_r * sigma.unsqueeze(0)): [d_out, r]
            #    @ Vh_r:                    [r, d_in]
            delta = (module.U_r * module.sigma.unsqueeze(0)) @ module.Vh_r
            # 2) final fused weight
            W_final = module.W_p + delta  # both are [d_out, d_in]

            # 3) new plain Linear
            out_f, in_f = W_final.shape
            new_lin = nn.Linear(in_features=in_f, out_features=out_f,
                                bias=(module.bias is not None))
            # 4) copy weights and bias
            new_lin.weight.data.copy_(W_final)
            if module.bias is not None:
                new_lin.bias.data.copy_(module.bias.data)

            # 5) swap it back into the parent
            parent_name, attr = name.rsplit(".", 1)
            parent_mod = name2mod[parent_name]
            setattr(parent_mod, attr, new_lin)

    return nudged_model

def train_imitation(
    nudged_model: nn.Module,
    dense_model: nn.Module,
    dataset,                # HF Dataset of dicts: pixel_values, input_ids, attention_mask
    device="cuda",
    lr=1e-3,
    epochs=5):

    # Freeze all parameters in nudged_model
    for p in nudged_model.parameters():
        p.requires_grad = False

    # then un‐freeze only your sigmas
    for layer in nudged_model.modules():
        if isinstance(layer, NudgedLinear):
            layer.sigma.requires_grad = True

    # make sure only sigma is trainable
    optimizer = torch.optim.Adam(
        [p for p in nudged_model.parameters() if p.requires_grad],
        lr=lr
    )

    loss_fn = nn.MSELoss()
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Start with nudged_model on CPU
    nudged_model.to("cpu")

    # Precompute dense outputs
    dense_model.to(device).eval()
    # Store dense logits on CPU
    all_dense_logits = []
    for traj in loader:
        inputs = {k:v.to(device) for k,v in traj.items()}
        with torch.no_grad():
            logits_dense = dense_model(**inputs)["logits"]  # [1, T, C]
        all_dense_logits.append(logits_dense.cpu())
    dense_model.to("cpu")
    torch.cuda.empty_cache()
    print("[*] Precomputed dense logits for all trajectories.")

    # Move nudged_model to GPU
    nudged_model.to(device)

    for epoch in range(epochs):
        total_loss = 0.0

        for idx, traj in enumerate(loader):
            # traj is e.g. {'pixel_values':Tensor[1,3,224,224], 'input_ids':Tensor[1,T], 'attention_mask':Tensor[1,T]}
            # unpack one‐sample batch
            inputs = {k: v.to(device) for k,v in traj.items() }

            optimizer.zero_grad()
                        
            # Run your nudged model on GPU (autograd enabled).
            out_nudged = nudged_model(**inputs)
            logits_nudged = out_nudged["logits"]      # Tensor on GPU

            logits_dense = all_dense_logits[idx].to(device).squeeze(0)  # [T,C]
            logits_nudged = logits_nudged.squeeze(0)  # [T,C]

            # Compute the summed “trajectory” loss and backprop
            loss = 0.0
            for t in range(logits_nudged.size(0)):
                loss = loss + loss_fn(logits_nudged[t], logits_dense[t])

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # Now drop everything from this iteration
            del inputs, out_nudged, logits_nudged, logits_dense, loss
            torch.cuda.empty_cache()
            

        print(f"Epoch {epoch}  avg traj loss = {total_loss/len(dataset):.4f}")

    return nudged_model


if __name__ == "__main__":

    # Load the pruned OpenVLA model
    print("[*] Loading pruned OpenVLA model...")
    path_to_pruned_model = "/workspace/models/openvla-7b-pruned-2_4-Wanda-pruned-full_model"
    cfg = DummyConfig(path_to_pruned_model)
    pruned_model = get_model(cfg)
    pruned_model = pruned_model.float()

    print("[*] Loading dense OpenVLA model...")
    path_to_dense_model = "/workspace/models/openvla-7b-finetuned-libero-spatial"
    cfg = DummyConfig(path_to_dense_model)
    dense_model = get_model(cfg)
    dense_model = dense_model.float()

    # build nudged_model with trainable singular‐values
    nudged_model = build_nudged_model(pruned_model, dense_model, rank=RANK)

    # Build the dataset
    tfrecord_dir = "/workspace/data/modified_libero_rlds/libero_spatial_no_noops/1.0.0"
    tfrecord_paths = [
        os.path.join(tfrecord_dir, f)
        for f in sorted(os.listdir(tfrecord_dir))
        if ".tfrecord" in f
    ]

    # Get the processor
    processor = get_openvla_processor(cfg)

    calib_data = build_calibration_dataset_from_examples(
        tfrecord_paths=tfrecord_paths,
        device="cuda",
        processor=processor,
        num_samples=NUM_DATA_SAMPLES,
    )

    print("Number of Calibration Samples", len(calib_data))

    calib_list = []
    for s in calib_data:
        calib_list.append({
            "pixel_values"  : s["pixel_values"].cpu(),      # bf16
            "input_ids"     : s["input_ids"].cpu(),         # int64
            "attention_mask": s["attention_mask"].cpu(),    # int64
        })

    ds = Dataset.from_list(calib_list).with_format("torch")
    print(len(ds), ds.column_names)        # sanity-check

    # ---- turn it into a HF Dataset -------------------------------------------
    ds = Dataset.from_list(calib_list)              # keeps torch tensors as-is
    ds = ds.with_format("torch")                    # no dtype conversion now

    # run the simple imitation‐gradient trainer
    nudged = train_imitation(
        nudged_model=nudged_model,
        dense_model=dense_model,
        dataset=ds,         # your torch‐formatted HF Dataset
        device="cuda",
        lr=LEARNING_RATE,
        epochs=EPOCHS
    )

    if save_with_custom_nudged_layers: 
        # Save your final nudged model
        nudged.save_pretrained(f"pruned_model_nudged_linear_rank_{RANK}_with_IL_custom_nuged_layers")
        print(f"Patched model saved to pruned_model_nudged_linear_rank_{RANK}_with_IL_custom_nuged_layers")
    else:
        # now export it back to plain nn.Linear’s
        nudged = export_to_dense_linear(nudged)
        nudged.save_pretrained(f"pruned_model_nudged_linear_rank_{RANK}_with_IL")
        print(f"Patched model saved to pruned_model_nudged_linear_rank_{RANK}_with_IL")

    pdb.set_trace()

