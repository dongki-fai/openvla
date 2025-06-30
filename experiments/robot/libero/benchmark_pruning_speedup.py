import torch
from torch.sparse import to_sparse_semi_structured
from torch.utils.benchmark import Timer
from torch import nn
from experiments.robot.openvla_utils import get_processor
from experiments.robot.robot_utils    import get_model
from PIL import Image

# Force CUTLASS for 2:4 sparsity
from torch.sparse import SparseSemiStructuredTensor
SparseSemiStructuredTensor._FORCE_CUTLASS = True

class DummyConfig:
    def __init__(self, pretrained_checkpoint="/workspace/models/openvla-7b-finetuned-libero-spatial"):
        self.model_family       = "openvla"
        self.pretrained_checkpoint = pretrained_checkpoint
        self.load_in_8bit      = False
        self.load_in_4bit      = False
        self.pruned_inference  = False
        self.load_to_cpu       = False

def bytes_for_dtype(dtype):
    """Return bytes per element for a given torch.dtype."""
    # adjust if using bfloat16 or other
    if dtype == torch.float32: return 4
    if dtype == torch.float16: return 2
    if dtype == torch.int8  : return 1
    raise ValueError(f"Unknown dtype {dtype}")

def compute_memory(
    W: torch.Tensor,
    rank: int,
    dtype=torch.float16
):
    """
    Returns a dict with memory usage (in megabytes) for:
      - dense W
      - 2:4 sparse version of W
      - low-rank hook storing (u, sigma, v) of given rank
    Assumes 2:4 mask: exactly half the entries of W are nonzero.
    """
    out_dim, in_dim = W.shape
    bytes_per = bytes_for_dtype(dtype)

    # dense
    dense_bytes = out_dim * in_dim * bytes_per

    # sparse (2:4 semi-structured)
    #    exactly half the weights are stored as values,
    #    plus one bit per weight for the mask => 0.125 bytes each.
    nnz = (out_dim * in_dim) // 2
    mask_bits = out_dim * in_dim   # one bit per entry
    sparse_bytes = nnz * bytes_per + mask_bits / 8

    # hook
    #    u: out_dim × rank
    #    sigma: rank
    #    v: rank × in_dim
    hook_bytes = (out_dim * rank + rank + rank * in_dim) * bytes_per

    return {
        "dense_MB":  dense_bytes  / (1024**2),
        "sparse_MB": sparse_bytes / (1024**2),
        "hook_MB":   hook_bytes   / (1024**2),
    }


def benchmark_layer(
    model, 
    layer_name, 
    enable_hook=False, 
    rank=1,
    device="cuda"
):
    """
    Benchmarks:
      - sparse-only: O(d^2/2)
      - optionally + low-rank hook of rank `rank`: O(d*r)
    If enable_hook=True, we compute the top-r SVD of the pruning gap on-the-fly.
    Returns a dict of timings in milliseconds.
    """
    # 1) extract the pruned layer and its weight
    target_layer = dict(model.named_modules())[layer_name]
    W = target_layer.weight.detach().to(torch.float32)
    print(f"Rank of W: {torch.linalg.matrix_rank(W)}")
    W = W.to(torch.float16)

    # dummy input x: shape (in_dim, batch=1)
    x = torch.rand(W.shape[1], W.shape[0]).half().cuda() 

    linear = torch.nn.Linear(W.shape[1], W.shape[0]).half().cuda()
    linear.weight = torch.nn.Parameter(W)

    with torch.inference_mode():
        dense_output = linear(x)
        dense_t = Timer(stmt="linear(x)",
                        globals={"linear": linear,
                                "x": x}).blocked_autorange().median * 1e3

        # accelerate via SparseSemiStructuredTensor
        linear.weight = torch.nn.Parameter(to_sparse_semi_structured(linear.weight))

        sparse_output = linear(x)
        sparse_t = Timer(stmt="linear(x)",
                        globals={"linear": linear,
                                "x": x}).blocked_autorange().median * 1e3

        if not enable_hook:
            return {"dense_t": dense_t, "sparse_t": sparse_t}


    # Prepare low-rank factors via fake SVD
    d_out, d_in = W.shape
    print(f"Shape of W: {W.shape} | d_out: {d_out}, d_in: {d_in}")
    
    # Fake singular vectors and values
    U_r  = torch.randn(d_out, rank, device=device).half()     # left vecs
    S_r  = torch.rand(rank, device=device).abs().half()       # singular values >0
    Vh_r = torch.randn(rank, d_in, device=device).half()      # right vecs

    # Define the hook for SVD approximation
    def hook(u, sigma, v, inp):
        s = v @ inp                       # (r, 1)
        s = sigma.unsqueeze(1) * s       # scale each component
        return u @ s     

    with torch.inference_mode():
        hook_t = Timer(
            stmt="hook(U_r, S_r, Vh_r, x)",
            globals={"hook": hook, "U_r": U_r, "S_r": S_r, "Vh_r": Vh_r, "x": x}
        ).blocked_autorange().median * 1e3  # ms

    total_t = sparse_t + hook_t


    mem = compute_memory(W, rank, dtype=W.dtype)
    print(f"Memory (MB): dense={mem['dense_MB']:.5f} sparse={mem['sparse_MB']:.5f} hook(r={rank})={mem['hook_MB']:.5f}")

    return {
        "dense_t": dense_t,
        "sparse_t": sparse_t,
        "hook_t":   hook_t,
        "total_t":  total_t,
        "rank":      rank
    }

if __name__ == "__main__":
    # 0) load pruned model + processor
    model_directory = "/workspace/models/openvla-7b-pruned-2_4-Wanda-pruned-language_backbone-ignore-lang-layers-0-15"
    cfg   = DummyConfig() #(pretrained_checkpoint=model_directory)
    model = get_model(cfg)
    model = model.float() 
    model = model.to("cuda") 

    layer = "language_model.model.layers.16.self_attn.q_proj"

    # sparse-only
    no_hook = benchmark_layer(model, layer, enable_hook=False)
    print(f"Dense: {no_hook['dense_t']:.3f}ms Sparse: {no_hook['sparse_t']:.3f}ms | Speedup: {(no_hook['dense_t'] / no_hook['sparse_t']):.3f}x")

    # with rank-1 hook
    r1 = benchmark_layer(model, layer, enable_hook=True, rank=1)
    print(f"Rank-1 hook: dense {r1['dense_t']:.3f}, Sparse {r1['sparse_t']:.3f} + hook {r1['hook_t']:.3f} = {r1['total_t']:.3f} ms | Speedup: {(r1['dense_t']/r1['total_t']):.3f}x")

    # with rank-8 hook
    r8 = benchmark_layer(model, layer, enable_hook=True, rank=8)
    print(f"Rank-8 hook: dense {r8['dense_t']:.3f}, Sparse {r8['sparse_t']:.3f} + hook {r8['hook_t']:.3f} = {r8['total_t']:.3f} ms | Speedup: {(r8['dense_t'] / r8['total_t']):.3f}x")

    # with rank-25 hook
    r25 = benchmark_layer(model, layer, enable_hook=True, rank=25)
    print(f"Rank-25 hook: dense {r25['dense_t']:.3f}, Sparse {r25['sparse_t']:.3f} + hook {r25['hook_t']:.3f} = {r25['total_t']:.3f} ms | Speedup: {(r25['dense_t'] / r25['total_t']):.3f}x")

    # with rank-50 hook
    r50 = benchmark_layer(model, layer, enable_hook=True, rank=50)
    print(f"Rank-50 hook: dense {r50['dense_t']:.3f}, Sparse {r50['sparse_t']:.3f} + hook {r50['hook_t']:.3f} = {r50['total_t']:.3f} ms | Speedup: {(r50['dense_t'] / r50['total_t']):.3f}x")

    # with rank-100 hook
    r100 = benchmark_layer(model, layer, enable_hook=True, rank=100)
    print(f"Rank-100 hook: dense {r100['dense_t']:.3f}, Sparse {r100['sparse_t']:.3f} + hook {r100['hook_t']:.3f} = {r100['total_t']:.3f} ms | Speedup: {(r100['dense_t'] / r100['total_t']):.3f}x")

    # with rank-200 hook
    r200 = benchmark_layer(model, layer, enable_hook=True, rank=200)
    print(f"Rank-200 hook: dense {r200['dense_t']:.3f}, Sparse {r200['sparse_t']:.3f} + hook {r200['hook_t']:.3f} = {r200['total_t']:.3f} ms | Speedup: {(r200['dense_t'] / r200['total_t']):.3f}x")

    # with rank-500 hook
    r500 = benchmark_layer(model, layer, enable_hook=True, rank=500)
    print(f"Rank-500 hook: dense {r500['dense_t']:.3f}, Sparse {r500['sparse_t']:.3f} + hook {r500['hook_t']:.3f} = {r500['total_t']:.3f} ms | Speedup: {(r500['dense_t'] / r500['total_t']):.3f}x")

    # with rank-1000 hook
    r1000 = benchmark_layer(model, layer, enable_hook=True, rank=1000)
    print(f"Rank-1000 hook: dense {r1000['dense_t']:.3f}, Sparse {r1000['sparse_t']:.3f} + hook {r1000['hook_t']:.3f} = {r1000['total_t']:.3f} ms | Speedup: {(r1000['dense_t'] / r1000['total_t']):.3f}x")