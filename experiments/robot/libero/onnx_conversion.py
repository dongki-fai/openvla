from transformers import AutoConfig, AutoModelForVision2Seq
import safetensors.torch
from safetensors.torch import load_file
import glob
import torch
from PIL import Image
import torch
import pdb

from experiments.robot.openvla_utils import get_processor


directory = "openvla-7b-pruned-2_4-disabled-sparse-compression"

# Manually load the raw compressed state dict (find all your shards)
shard_paths = sorted(glob.glob(directory+"/model-*-of-*.safetensors"))

# Load the safetensors weights (float32)
state_dict = {}
for p in shard_paths:
    sd = load_file(p)
    # Convert each tensor to bfloat16
    for k, v in sd.items():
        # print("k:", k)
        state_dict[k] = v.to(torch.float16)

# Instantiate bare model with no quant hooks
config = AutoConfig.from_pretrained(directory, trust_remote_code=True)
model = AutoModelForVision2Seq.from_config(
    config,
    torch_dtype=torch.float16,
    trust_remote_code=True,
)
model.load_state_dict(state_dict, strict=False)
model.to("cuda")

# Setup dummy config with checkpoint
class DummyConfig:
    model_family = "openvla"
    pretrained_checkpoint = "/workspace/models/openvla-7b-finetuned-libero-spatial"
    load_in_8bit = False
    load_in_4bit = False

cfg = DummyConfig()

# Get the processor
processor = get_processor(cfg)

# construct a dummy observation
dummy_img = Image.new("RGB", (256,256), color="gray")
prompt   = "What action should the robot take to pick up the black bowl?"

# preprocess via your HF processor
#    (this will give you pixel_values, input_ids, attention_mask)
inputs = processor(
    prompt,
    dummy_img,
    return_tensors="pt",
).to("cuda", dtype=torch.float16)


# Select inputs relevant to model forward
input_names = list(inputs.keys())
input_values = tuple(inputs[k] for k in input_names)

# Create export path
onnx_path = "openvla_2_4_pruned.onnx"

# Make sure model is in eval mode
model.eval()

class GenerationWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, pixel_values):
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            max_new_tokens=7 
        )

try: 
    wrapped = GenerationWrapper(model).to("cuda").eval()

    with torch.no_grad():
        torch.onnx.export(
            wrapped,
            input_values,
            "openvla_action_generator.onnx",
            input_names=input_names,
            output_names=["action_token_ids"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "seq_len"},
                "attention_mask": {0: "batch_size", 1: "seq_len"},
                "pixel_values": {0: "batch_size"},
            },
            opset_version=17
    )
except Exception as e:
    print(f"Error during ONNX export: {e}")

print(f"Exported ONNX model to: {onnx_path}")




pdb.set_trace()

