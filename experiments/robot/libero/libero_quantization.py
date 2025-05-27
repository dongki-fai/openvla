from transformers import AutoModelForCausalLM
from experiments.robot.robot_utils import get_model
from transformers import AutoProcessor
from experiments.robot.openvla_utils import get_processor

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

# Setup dummy config with checkpoint
class DummyConfig:
    model_family = "openvla"
    pretrained_checkpoint = "/workspace/models/openvla-7b-finetuned-libero-spatial"
    load_in_8bit = False
    load_in_4bit = False

def build_llm_inputs(model, processor, image, instruction):
    inputs = processor(images=image, text=instruction, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(model.device, dtype=torch.bfloat16)
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)

    with torch.no_grad():
        # Encode image and project to LLM input dim
        visual_embeds = model.vision_backbone(pixel_values)
        projected_visual_embeds = model.projector(visual_embeds)

        # Token embeddings from LLM
        inputs_embeds = model.language_model.model.embed_tokens(input_ids)

        # Fuse: prepend image tokens to text tokens
        fused = torch.cat([projected_visual_embeds, inputs_embeds], dim=1)

        # Fix attention mask to match fused length
        B, V, _ = projected_visual_embeds.shape
        V_mask = torch.ones((B, V), dtype=attention_mask.dtype, device=attention_mask.device)
        fused_attention_mask = torch.cat([V_mask, attention_mask], dim=1)

    return {
        "inputs_embeds": fused.squeeze(0),
        "attention_mask": fused_attention_mask.squeeze(0),
    }

def decode_jpeg(jpeg_bytes):
    """Decode JPEG bytes into RGB image as np.uint8 array."""
    image = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
    return image


def build_calibration_dataset_from_examples(
    tfrecord_paths,
    model,
    processor,
    num_samples=16,
    seed=42,
):
    calibration_set = []
    random.seed(seed)

    for tfrecord_path in tfrecord_paths:
        print(f"[*] Scanning: {tfrecord_path}")
        dataset = tf.data.TFRecordDataset(tfrecord_path)

        for record in dataset:
            if len(calibration_set) >= num_samples:
                break

            try:
                example = tf.train.Example()
                example.ParseFromString(record.numpy())

                instruction = example.features.feature["steps/language_instruction"].bytes_list.value[0].decode()
                img_bytes = example.features.feature["steps/observation/image"].bytes_list.value[0]
                image = decode_jpeg(img_bytes)

                result = build_llm_inputs(model, processor, image, instruction)
                calibration_set.append(result)

            except Exception as e:
                print(f"[!] Skipping record due to error: {e}")
                continue

        if len(calibration_set) >= num_samples:
            break

    print(f"[✓] Collected {len(calibration_set)} calibration examples.")
    return calibration_set


cfg = DummyConfig()

# Step 2: Load the full OpenVLA model
print("[*] Loading full OpenVLA model...")
model = get_model(cfg)

# print(f"\n[Model Class] {type(model)}")
# print("\n[Top-level Attributes and Submodules]")

# for name, module in model.named_children():
#     print(f" - {name}: {type(module)}")

# Extract the LLM backbone
# llm = model.language_model
llm = model.language_model.float() 
# Get the processor
processor = get_processor(cfg)

# Dummy test
dummy_image = Image.new("RGB", (256, 256), color="gray")
dummy_instruction = "Pick up the black bowl and place it on the tray."

inputs = build_llm_inputs(model, processor, dummy_image, dummy_instruction)

# 256 vision tokens from the image encoding
# 15 text tokens from the instruction

print("✓ inputs_embeds:", inputs["inputs_embeds"].shape)
print("✓ attention_mask:", inputs["attention_mask"].shape)

tfrecord_dir = "/workspace/data/modified_libero_rlds/libero_spatial_no_noops/1.0.0"
tfrecord_paths = [
    os.path.join(tfrecord_dir, f)
    for f in sorted(os.listdir(tfrecord_dir))
    if ".tfrecord" in f
]

calib_data = build_calibration_dataset_from_examples(
    tfrecord_paths=tfrecord_paths,
    model=model,
    processor=processor,
    num_samples=40,
)

print("Number of Calibration Samples", len(calib_data))


# ---- build the list with *tensors* already in the right dtype -------------
calib_list = []
for s in calib_data:
    calib_list.append({
        "inputs_embeds" : s["inputs_embeds"].to(torch.float16).cpu(),  # tensor bf16
        "attention_mask": s["attention_mask"].to(torch.int64 ).cpu(),   # tensor i64
    })

# ---- turn it into a HF Dataset -------------------------------------------
ds = Dataset.from_list(calib_list)              # keeps torch tensors as-is
ds = ds.with_format("torch")                    # no dtype conversion now

# quick sanity check --------------------------------------------------------
print("N samples :", len(ds))
print("keys      :", ds.column_names)
print("embed one :", ds[0]["inputs_embeds"].dtype, ds[0]["inputs_embeds"].shape)
print("mask  one :", ds[0]["attention_mask"].dtype, ds[0]["attention_mask"].shape)


recipe = GPTQModifier(
    scheme      = "W4A16",    # int-4 weights, fp16 activations
    targets     = "Linear",   # all nn.Linear layers
    ignore      = ["lm_head"],# keep LM head in fp16
    group_size  = 128,        # “G128” layout
)

oneshot(
    model=llm,            # ← just the language backbone
    dataset=ds,           # ← HF Dataset we just built
    recipe=recipe,
    processor=processor,    # <-- add this line
    num_calibration_samples=len(ds),
    max_seq_length=350,   # length of your fused embed sequence
    output_dir="quantizedtest"
)

# put the quantised backbone back
model.language_model = llm
model.eval()

# 2save everything in one go
save_dir = "/workspace/models/openvla-7b-W4A16-G128-full"
model.save_pretrained(save_dir, safe_serialization=True, save_compressed=True)
processor.save_pretrained(save_dir)
print("✔ saved to", save_dir)