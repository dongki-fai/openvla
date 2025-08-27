########################################################################
# Trying to load the model in but unsupported compression from transformers
########################################################################

# from transformers import AutoModelForVision2Seq

# vla = AutoModelForVision2Seq.from_pretrained(
#         "openvla-7b-pruned-2_4-ct",
#         attn_implementation="eager",
#         torch_dtype="bfloat16",
#         trust_remote_code=True)


########################################################################
#  Trying to load the model but pretend no compression has been applied
########################################################################
# from transformers import AutoModelForVision2Seq, AutoConfig
# import torch 

# config = AutoConfig.from_pretrained(
#     "openvla-7b-pruned-2_4-ct",
#     trust_remote_code=True
# )

# # strip out any quantization bits
# cfg_dict = config.to_dict()
# cfg_dict.pop("quantization_config", None)           # REMOVE it entirely
# config = type(config).from_dict(cfg_dict)           # re-build a “clean” config

# # load your pruned OpenVLA
# vla = AutoModelForVision2Seq.from_pretrained(
#     "openvla-7b-pruned-2_4-ct",
#     config=config,
#     attn_implementation="eager",
#     torch_dtype="bfloat16",
#     trust_remote_code=True,
#     low_cpu_mem_usage=True,
# )

########################################################################
#  Trying to load the model with deepspeed
########################################################################
import deepspeed
from transformers import AutoConfig, AutoModelForVision2Seq
import safetensors.torch
from safetensors.torch import load_file
import glob
import torch
from PIL import Image
import torch

from experiments.robot.openvla_utils import get_openvla_processor


# Manually load the raw compressed state dict
# Find all your shards
shard_paths = sorted(glob.glob("openvla-7b-pruned-2_4-ct-sandbox/model-*-of-*.safetensors"))

# Load and merge
state_dict = {}
for p in shard_paths:
    sd = load_file(p)
    state_dict.update(sd)

# Instantiate bare model with no quant hooks
config = AutoConfig.from_pretrained("openvla-7b-pruned-2_4-ct-sandbox", trust_remote_code=True)
model = AutoModelForVision2Seq.from_config(
    config,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
model.load_state_dict(state_dict, strict=False)

# Wrap in DeepSpeed’s sparse inference engine
engine = deepspeed.init_inference(
    model,
    mp_size=1,
    dtype=torch.bfloat16,
    replace_method="auto",
    replace_with_kernel_inject=True,
)

# Setup dummy config with checkpoint
class DummyConfig:
    model_family = "openvla"
    pretrained_checkpoint = "/workspace/models/openvla-7b-finetuned-libero-spatial"
    load_in_8bit = False
    load_in_4bit = False

cfg = DummyConfig()

# Get the processor
processor = get_openvla_processor(cfg)

# construct a dummy observation
dummy_img = Image.new("RGB", (256,256), color="gray")
prompt   = "What action should the robot take to pick up the black bowl?"

# preprocess via your HF processor
#    (this will give you pixel_values, input_ids, attention_mask)
inputs = processor(
    prompt,
    dummy_img,
    return_tensors="pt",
)

# get the device from the wrapped model inside DeepSpeed
device = next(engine.module.parameters()).device

# then move & cast your inputs appropriately
for k, v in inputs.items():
    # move every tensor to the same device as the model
    v = v.to(device)
    # cast only the pixel_values into bfloat16
    if k == "pixel_values":
        v = v.to(torch.bfloat16)
    inputs[k] = v

# run the forward pass
engine.module.config.use_cache = True      # turn it off by default
engine.eval()

for _ in range(10):
    import time
    st = time.time()
    with torch.no_grad():
        # if you’re using the built-in predict_action helper
        action = engine.module.predict_action(**inputs, unnorm_key="libero_spatial",do_sample=False, use_cache=True)
        # or, if you just want raw logits:
        # out = engine(**inputs)
        # logits = out.logits
    et = time.time()
    print("time:", et - st)

print("Predicted action:", action)


