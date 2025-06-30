import torch
from torch.sparse import to_sparse_semi_structured, SparseSemiStructuredTensor
from torch.utils.benchmark import Timer
from transformers import AutoModelForVision2Seq, AutoConfig
import glob
from safetensors.torch import load_file
# import pdb
from PIL import Image
from torch import nn

from experiments.robot.openvla_utils import get_processor
from experiments.robot.robot_utils import get_model

SparseSemiStructuredTensor._FORCE_CUTLASS = True

class DummyConfig():
    def __init__(self, pretrained_checkpoint="/workspace/models/openvla-7b-finetuned-libero-spatial"):
        self.model_family = "openvla"
        self.pretrained_checkpoint = pretrained_checkpoint
        self.load_in_8bit = False
        self.load_in_4bit = False
        self.pruned_inference = False
        self.load_to_cpu = False

model_directory = "/workspace/models/openvla-7b-pruned-2_4-Wanda-pruned-language_backbone-ignore-lang-layers-0-15"
cfg = DummyConfig()

# Load the full OpenVLA model
print("[*] Loading full OpenVLA model...")
model = get_model(cfg)
model = model.float()  

# Get the processor
processor = get_processor(cfg)

# construct a dummy observation
dummy_img = Image.new("RGB", (256,256), color="gray")
prompt   = "What action should the robot take to pick up the black bowl?"

# preprocess via your HF processor
inputs = processor(prompt, dummy_img).to("cuda", dtype=torch.bfloat16)


# ##########

# ### Languauge Model Linear Layer Benchmarking

# ##########

# Pick a specific linear layer for testing
target_layer = dict(model.named_modules())["language_model.model.layers.30.self_attn.q_proj"]
W = target_layer.weight.detach().to(torch.float16)

x = torch.rand(W.shape[1], W.shape[0]).half().cuda()  # [in_features, out_features]

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

    # sparse and dense matmul are numerically equivalent
    # On an A100 80GB, we see: `Dense: 0.870ms Sparse: 0.630ms | Speedup: 1.382x`
    # assert torch.allclose(sparse_output, dense_output, atol=1e-3)
    print(f"Dense: {dense_t:.3f}ms Sparse: {sparse_t:.3f}ms | Speedup: {(dense_t / sparse_t):.3f}x")


# pdb.set_trace()

# # mask Linear weight to be 2:4 sparse
# mask = torch.Tensor([0, 0, 1, 1]).tile((3072, 2560)).cuda().bool()
# linear = torch.nn.Linear(10240, 3072).half().cuda().eval()
# linear.weight = torch.nn.Parameter(mask * linear.weight)

# # linear.weight.data[:, 0] = 1.0

# # # Print the first 20 by 20 block of the weight matrix
# # print(linear.weight[:10, :10])

# # # Print the shape of the weight matrix 
# # print(f"Weight shape: {linear.weight.shape}")

# x = torch.rand(3072, 10240).half().cuda()




# with torch.inference_mode():

#     # Warm-up
#     _ = model.predict_action(**inputs, unnorm_key="libero_spatial", do_sample=False)

#     dense_t = Timer(
#         stmt="model.predict_action(**inputs, unnorm_key='libero_spatial', do_sample=False)",
#         globals={"model": model, "inputs": inputs}
#     ).blocked_autorange().median * 1e3

#     for fqn, module in model.named_modules():
#         if isinstance(module, nn.Linear): #and "layer" in fqn:
#             if module.weight.shape[0] % 32 == 0 and module.weight.shape[1] % 64 == 0:
#                 print(f"Converting {fqn} to sparse semi-structured")
#                 module.weight = torch.nn.Parameter(to_sparse_semi_structured(module.weight))

#             else:
#                 print(f"[Dimension Error] Skipping {fqn} as it does not have the right dimensions for sparse semi-structured conversion.")
#         else:
#             print(f"Skipping {fqn} as it is not a linear layer in the transformer blocks.")

#     _ = model.predict_action(**inputs, unnorm_key="libero_spatial", do_sample=False)

#     sparse_t = Timer(
#         stmt="model.predict_action(**inputs, unnorm_key='libero_spatial', do_sample=False)",
#         globals={"model": model, "inputs": inputs}
#     ).blocked_autorange().median * 1e3

#     # sparse and dense matmul are numerically equivalent
#     # On an A100 80GB, we see: `Dense: 0.870ms Sparse: 0.630ms | Speedup: 1.382x`
#     # assert torch.allclose(sparse_output, dense_output, atol=1e-3)
#     print(f"Dense: {dense_t:.3f}ms Sparse: {sparse_t:.3f}ms | Speedup: {(dense_t / sparse_t):.3f}x")


