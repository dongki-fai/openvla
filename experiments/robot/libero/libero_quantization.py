import io
import tensorflow as tf

from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor
from optimum.gptq import GPTQQuantizer


def load_tfrecord_inputs(data_path, processor, n=16):
    print(f"[*] Loading {n} samples from LIBERO TFRecord for GPTQ calibration...")

    raw_dataset = tf.data.TFRecordDataset([data_path])
    examples = []

    for raw_record in raw_dataset.take(n):
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())

        image_bytes = example.features.feature["steps/observation/image"].bytes_list.value[0]
        instruction = example.features.feature["steps/language_instruction"].bytes_list.value[0].decode("utf-8")

        # Fix: wrap image_bytes in BytesIO before passing to PIL
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        processed = processor(images=image, text=instruction, return_tensors="pt")
        examples.append(processed)

    return examples

# Load your full model (no quantization)
model_id = "/home/jason/Desktop/Jason/FieldAI/VLA/models/openvla-7b-finetuned-libero-spatial"
model = AutoModelForVision2Seq.from_pretrained(model_id, 
                                                device_map="auto",
                                                trust_remote_code=True)
processor = AutoProcessor.from_pretrained(model_id)

# Load calibration data (images + text)
calibration_datapath = '/home/jason/Desktop/Jason/FieldAI/VLA/modified_libero_rlds/libero_spatial_no_noops/1.0.0/libero_spatial-train.tfrecord-00000-of-00016'
 # List of {"input_ids", "pixel_values"} or similar
calibration_dataset =  load_tfrecord_inputs(calibration_datapath, processor, n=16)

# Quantize the model
quantizer = GPTQQuantizer(
    bits=4,
    dataset=calibration_dataset,
    block_name_to_quantize="language_model.model.layers",
    batch_size=1  # Use smallest batch size to minimize memory
)

quantized_model = quantizer.quantize_model(model)

# Save it
quantized_model.save_pretrained("./openvla-7b-libero-gptq")
processor.save_pretrained("./openvla-7b-libero-gptq")
