import torch
import torch.nn.functional as F
from torch.sparse import to_sparse_semi_structured
from torch.utils.benchmark import Timer
from torch.profiler import profile, ProfilerActivity, record_function
from torch import nn
from torch import Tensor
from experiments.robot.openvla_utils import get_processor
from experiments.robot.robot_utils    import get_model
from PIL import Image

import torch._dynamo as dynamo
import tempfile
import os
import shutil
import gc
from contextlib import contextmanager
from statistics import mean, stdev

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

def get_cuda_time(prof, name):
    for evt in prof.key_averages():
        if evt.key == name:
            return evt.cuda_time_total
    return 0.0

def reset_compile_state():
    dynamo.reset()  # clears in-process compile caches
    # give Inductor/Triton fresh on-disk caches so nothing is reused
    for var in ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR"):
        old = os.environ.get(var)
        new = tempfile.mkdtemp(prefix=var.lower()+"_")
        os.environ[var] = new
        if old: shutil.rmtree(old, ignore_errors=True)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

@contextmanager
def cuda_peak_memory():
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    try:
        yield
    finally:
        torch.cuda.synchronize()
        peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
        print(f"peak_alloc_MB: {peak_mb:.2f}")

def benchmark_layer(
    model,
    layer_name,
    enable_hook=False, 
    rank=1,
    device="cuda",
    batch_size = 4096
):

    # Extract the pruned layer and its weight
    target_layer = dict(model.named_modules())[layer_name]
    W = target_layer.weight.detach().to(torch.float32)
    W = W.half().cuda()

    d_out, d_in = W.shape
    batch_size = batch_size
    print(f"Shape of W: {W.shape} | d_out: {d_out}, d_in: {d_in}")

    # dummy input x: shape (in_dim, batch=1)
    x = torch.rand(batch_size, d_in).half().cuda() 

    linear = torch.nn.Linear(d_in, d_out).half().cuda()
    linear.weight = torch.nn.Parameter(W)

    with torch.inference_mode():

        dense_output = linear(x)
        time = Timer(stmt="linear(x)",
                        globals={"linear": linear,
                                "x": x}).blocked_autorange(min_run_time=4.0)

        per_run_s = [t / time.number_per_run for t in time.raw_times]
        dense_mean_ms   = mean(per_run_s) * 1e3
        dense_stdev_ms  = stdev(per_run_s) * 1e3

        # accelerate via SparseSemiStructuredTensor
        linear.weight = torch.nn.Parameter(to_sparse_semi_structured(linear.weight))
        
        # Fake singular vectors and values
        U_r  = torch.randn(d_out, rank, device=device).half()     
        S_r  = torch.rand(rank, device=device).abs().half()       
        Vh_r = torch.randn(rank, d_in, device=device).half()    

        U_scaled = U_r * S_r.unsqueeze(0)           
        V = Vh_r.t().contiguous()               
        U_T = U_scaled.t().contiguous()             

        @torch.jit.script
        def lowrank_jit(V: torch.Tensor,       
                        U_T: torch.Tensor,     
                        x: torch.Tensor):      
            # y = linear(x) + (x @ V) @ U_T
            # Fuse matmul + add via addmm (y += xV @ U_T)
            return torch.addmm(linear(x), x @ V, U_T, beta=1.0, alpha=1.0)

        lowrank_output = lowrank_jit(V, U_T, x)

        time = Timer(
            "lowrank_jit(V, U_T, x)",
            globals={"lowrank_jit": lowrank_jit, "V": V, "U_T": U_T, "x": x}
        ).blocked_autorange(min_run_time=4.0)

        per_run_s = [t / time.number_per_run for t in time.raw_times]
        fused_mean_ms   = mean(per_run_s) * 1e3
        fused_stdev_ms  = stdev(per_run_s) * 1e3

        print(f"Rank: {rank}, Dense Time {dense_mean_ms:.3f}ms ± {dense_stdev_ms:.3f}ms Fused Time: {fused_mean_ms:.3f}ms ± {fused_stdev_ms:.3f}ms | Fused Speedup: {dense_mean_ms / fused_mean_ms:.3f}x")


def profile_dense_layer(linear: nn.Module, x: Tensor) -> Tensor:
    return linear(x)

def profile_sparse_layer(linear: nn.Module, x: Tensor) -> Tensor:
    return linear(x)

def profile_sparse_and_SVD_layer(linear: nn.Module, x: Tensor, V: Tensor, U_T: Tensor) -> Tensor:
    # ys = x @ V
    # ys = ys @ U_T
    # return linear(x).add(ys)
    return torch.addmm(linear(x), x, U_T, beta=1.0, alpha=1.0)


class DenseWrap(nn.Module):
    def __init__(self, linear): 
        super().__init__()
        self.linear = linear
    def forward(self, x): 
        return self.linear(x)

class SparseOnlyWrap(nn.Module):
    def __init__(self, linear): 
        super().__init__()
        self.linear = linear
    def forward(self, x): 
        return self.linear(x)  # weight already 2:4

class SparseSVDFusedWrap(nn.Module):
    def __init__(self, linear, V, U_T):
        super().__init__()
        self.linear = linear
        self.register_buffer("V", V.contiguous())
        self.register_buffer("U_T", U_T.contiguous())
    def forward(self, x):
        y = F.linear(x, self.linear.weight, self.linear.bias)
        tmp = (x @ self.V) @ self.U_T
        # y.add_(tmp)
        return torch.add(y, tmp)  
    
def run_torch_profiling(model, layer_name, mode, rank=1, device="cuda", batch_size=400):

    reset_compile_state()

    # Extract the pruned layer and its weight
    target_layer = dict(model.named_modules())[layer_name]
    W = target_layer.weight.detach().to(torch.float32)
    W = W.half().cuda()

    d_out, d_in = W.shape
    batch_size = batch_size
    # print(f"Shape of W: {W.shape} | d_out: {d_out}, d_in: {d_in}")

    # dummy input x
    x = torch.rand(batch_size, d_in).half().cuda() 

    linear = torch.nn.Linear(d_in, d_out).half().cuda()
    linear.weight = torch.nn.Parameter(W)

    if mode == "dense":
        # Instantiate compiled modules (first call will compile; warm up once and exclude it from timing)
        dense_mod = torch.compile(DenseWrap(linear), mode="max-autotune")

        ## ------------------------------- Dense Layer Profiling -------------------------------
        ## -- Warm Up --
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True) as prof:
            with record_function("profile_dense_layer"):
                with torch.inference_mode():
                    # profile_dense_layer(linear, x)
                    dense_mod(x)
        
        # Dense layer profiling
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True) as prof_dense:
            with record_function("profile_dense_layer"):
                with torch.inference_mode():
                    dense_mod(x)
        
        print(prof_dense.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        prof_dense.export_chrome_trace("dense_trace.json")

        with cuda_peak_memory():
            _ = dense_mod(x)

        return get_cuda_time(prof_dense, "profile_dense_layer")
    
    if mode == "sparse":
        ## ------------------------------- Sparse Layer Profiling -------------------------------
        
        # accelerate via SparseSemiStructuredTensor
        linear.weight = torch.nn.Parameter(to_sparse_semi_structured(linear.weight))
        sparse_mod  = torch.compile(SparseOnlyWrap(linear), mode="max-autotune")

        ## -- Warm Up --
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True) as prof:
            with record_function("profile_sparse_layer"):
                with torch.inference_mode():
                    sparse_mod(x)

        # Sparse layer profiling
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True) as prof_sparse:
            with record_function("profile_sparse_layer"):
                with torch.inference_mode():
                    sparse_mod(x)

        print(prof_sparse.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        prof_sparse.export_chrome_trace("sparse_trace.json")

        with cuda_peak_memory():
            _ = sparse_mod(x)

        return get_cuda_time(prof_sparse, "profile_sparse_layer")

    if mode == "sparse_and_svd":

        ## ------------------------------- Sparse + SVD Layer Profiling -------------------------------
        
        # accelerate via SparseSemiStructuredTensor
        linear.weight = torch.nn.Parameter(to_sparse_semi_structured(linear.weight))

        # Random singular vectors and values
        U_r  = torch.randn(d_out, rank, device=device).half()     
        S_r  = torch.rand(rank, device=device).abs().half()       
        Vh_r = torch.randn(rank, d_in, device=device).half()    

        U_scaled = U_r * S_r.unsqueeze(0)           
        V = Vh_r.t().contiguous()               
        U_T = U_scaled.t().contiguous()   

        svd_mod = torch.compile(SparseSVDFusedWrap(linear, V, U_T), mode="max-autotune")

        ## -- Warm Up --
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True) as prof:
            with record_function("profile_sparse_and_SVD_layer"):
                with torch.inference_mode():
                    svd_mod(x)

        # Sparse + SVD layer profiling
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True) as prof_svd_sparse:
            with record_function("profile_sparse_and_SVD_layer"):
                with torch.inference_mode():
                    svd_mod(x)

        print(prof_svd_sparse.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        prof_svd_sparse.export_chrome_trace("sparse_svd_trace.json")


        with cuda_peak_memory():
            _ = svd_mod(x)

        return get_cuda_time(prof_svd_sparse, "profile_sparse_and_SVD_layer")

if __name__ == "__main__":
    # 0) load pruned model + processor
    model_directory = "/workspace/models/openvla-7b-GOAL-pruned-2_4-Wanda-pruned-language_backbone-15k-ORIGINAL"
    cfg   = DummyConfig() #(pretrained_checkpoint=model_directory)
    model = get_model(cfg)
    model = model.float() 
    model = model.to("cuda") 

    # layer = "language_model.model.layers.0.self_attn.q_proj"
    layer = "language_model.model.layers.14.self_attn.v_proj"
    batch_size = 279
    ranks = [1, 8, 25, 50, 100, 200, 300, 400, 500, 1000]
    trials = 10
    benchmarking_report = {}

    # ----- Run profiling for Dense Layer ---
    dense_time_lst = []
    for trail in range(trials):
        gpu_time = run_torch_profiling(model, layer, mode="dense", rank=None, device="cuda", batch_size=batch_size)
        dense_time_lst.append(gpu_time)

    avg_dense_time = sum(dense_time_lst) / trials
    stdev_dense_time = stdev(dense_time_lst) if len(dense_time_lst) > 1 else 0.0
    # ------------------------------------------

    # ----- Run Profiling for Sparse Layer ----
    sparse_time_lst = []
    for trial in range(trials):
        gpu_time = run_torch_profiling(model, layer, mode="sparse", rank=None, device="cuda", batch_size=batch_size)
        sparse_time_lst.append(gpu_time)

    avg_sparse_time = sum(sparse_time_lst) / trials
    stdev_sparse_time = stdev(sparse_time_lst) if len(sparse_time_lst) > 1 else 0.0
    # ------------------------------------------

    # ----- Run Profiling for Sparse + SVD Layer ----
    for rank in ranks:
        sparse_svd_time_lst = []
        for trial in range(trials):
            gpu_time = run_torch_profiling(model, layer, mode="sparse_and_svd", rank=rank, device="cuda", batch_size=batch_size)
            sparse_svd_time_lst.append(gpu_time)

        benchmarking_report[rank] = {
            "avg_sparse_svd_time": sum(sparse_svd_time_lst) / trials,
            "stdev_sparse_svd_time": stdev(sparse_svd_time_lst) if len(sparse_svd_time_lst) > 1 else 0.0
        }
    # ---------------------------------------------
    
    print("[*] Benchmarking Report:")
    print(f"Avg Dense Time: {avg_dense_time:.3f} ± {stdev_dense_time:.3f} us")
    print(f"Avg Sparse Time: {avg_sparse_time:.3f} ± {stdev_sparse_time:.3f} us")
    for rank, times in benchmarking_report.items():
        avg_sparse_svd_time = times["avg_sparse_svd_time"]
        stdev_sparse_svd_time = times["stdev_sparse_svd_time"]
        # Print the results
        print(f"Rank: {rank}, Avg Sparse + SVD Time: {avg_sparse_svd_time:.3f} ± {stdev_sparse_svd_time:.3f} us | Speedup: {avg_dense_time / avg_sparse_svd_time:.3f}x")
