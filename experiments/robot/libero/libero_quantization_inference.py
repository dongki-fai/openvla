from transformers import AutoModelForCausalLM
from experiments.robot.robot_utils import get_model
from transformers import AutoProcessor
from experiments.robot.openvla_utils import get_openvla_processor

import torch
from PIL import Image
import io
import os
import random
import tensorflow as tf
from transformers import AutoModelForCausalLM

if not hasattr(torch, "OutOfMemoryError"):
    class _OOM(RuntimeError): pass
    torch.OutOfMemoryError = _OOM

from datasets import IterableDataset, DatasetDict, Features, Sequence, Value, Array2D, DatasetInfo
from datasets import Dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier

FULL_MODEL_DIR = "/workspace/models/openvla-7b-finetuned-libero-spatial"
GPTQ_DIR    = "/workspace/models/openvla-7b-llm-W4A16_oneshot_llm_backbone"

# Setup dummy config with checkpoint
class DummyConfig:
    model_family = "openvla"
    pretrained_checkpoint = FULL_MODEL_DIR
    load_in_8bit = False
    load_in_4bit = False

cfg = DummyConfig()

# Step 2: Load the full OpenVLA model
print("[*] Loading full OpenVLA model...")
model = get_model(cfg)

processor = get_openvla_processor(cfg)


quant_llm = AutoModelForCausalLM.from_pretrained(
    GPTQ_DIR,
    torch_dtype="auto",        # reads fp16  activations
    attn_implementation="eager",
    low_cpu_mem_usage=False,   # <-- avoids the int-tensor grad bug
    trust_remote_code=True,
    _fast_init=False,            # <-- skip meta-tensor path

)

quant_llm.eval().to(model.device)

# stitch
model.language_model = quant_llm         # hot-swap
model.eval()


# # run the usual helper for a dummy obs 
# dummy_rgb = np.zeros((480, 640, 3), dtype=np.uint8)  
# obs       = {"full_image": dummy_rgb}
# task      = "pick up the black bowl and place it on the tray"

# with torch.no_grad():
#     action = get_vla_action(wrapper, processor,
#                             cfg.pretrained_checkpoint,
#                             obs, task,
#                             cfg.unnorm_key, cfg.center_crop)

# print("predicted 7-DoF action:", np.round(action, 3))