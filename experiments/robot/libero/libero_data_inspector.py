import tensorflow as tf
import io
from PIL import Image
import numpy as np
import os
import imageio
import random

def decode_jpeg(jpeg_bytes):
    """Decode JPEG bytes into a numpy uint8 RGB image with shape (256, 256, 3)."""
    image = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
    return np.asarray(image, dtype=np.uint8)

def save_mp4_video(frames, output_path, fps=30):
    """Saves a list of RGB frames to an MP4 file using imageio."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    writer = imageio.get_writer(output_path, fps=fps)
    for img in frames:
        writer.append_data(img)
    writer.close()
    print(f"[Success] Saved video to: {output_path}")

# === Setup ===
tfrecord_path = "/workspace/data/modified_libero_rlds/libero_spatial_no_noops/1.0.0/libero_spatial-train.tfrecord-00007-of-00016"
output_dir = "visualize_data"
os.makedirs(output_dir, exist_ok=True)
log_file_path = os.path.join(output_dir, "episode_language_log.txt")
log_file = open(log_file_path, "w")

# === Load all episodes into a list first ===
raw_dataset = list(tf.data.TFRecordDataset(tfrecord_path))
total_episodes = len(raw_dataset)
print(f"Total episodes in file: {total_episodes}")

# === Sample 10 random episode indices ===
sampled_indices = random.sample(range(total_episodes), min(25, total_episodes))

# === Process sampled episodes ===
for episode_idx in sampled_indices:
    raw_record = raw_dataset[episode_idx]
    example = tf.train.SequenceExample()
    example.ParseFromString(raw_record.numpy())
    context = example.context.feature

    num_steps = len(context["steps/reward"].float_list.value)
    print(f"\n=== Episode {episode_idx} ({num_steps} steps) ===")

    task_description = context["steps/language_instruction"].bytes_list.value[0].decode()
    file_path = context["episode_metadata/file_path"].bytes_list.value[0].decode()

    main_frames = []
    wrist_frames = []

    for t in range(num_steps):
        reward = context["steps/reward"].float_list.value[t]
        action = context["steps/action"].float_list.value[t*7:(t+1)*7]
        state = context["steps/observation/state"].float_list.value[t*8:(t+1)*8]
        joint = context["steps/observation/joint_state"].float_list.value[t*7:(t+1)*7]
        img_bytes = context["steps/observation/image"].bytes_list.value[t]
        wrist_bytes = context["steps/observation/wrist_image"].bytes_list.value[t]
        is_first = context["steps/is_first"].int64_list.value[t]
        is_last = context["steps/is_last"].int64_list.value[t]
        is_terminal = context["steps/is_terminal"].int64_list.value[t]
        discount = context["steps/discount"].float_list.value[t]

        # Decode JPEGs
        main_img = decode_jpeg(img_bytes)
        wrist_img = decode_jpeg(wrist_bytes)

        main_frames.append(main_img)
        wrist_frames.append(wrist_img)

        # print(f"Step {t}:")
        # print(f"  reward: {reward}")
        # print(f"  action: {action}")
        # print(f"  joint: {joint}")
        # print(f"  state: {state}")
        # print(f"  is_first: {is_first}")
        # print(f"  is_last: {is_last}")
        # print(f"  is_terminal: {is_terminal}")
        # print(f"  discount: {discount}")
        # print(f"  language_instruction: {task_description}")
        # print(f"  file_path: {file_path}")
        # print(f"  image shape: {main_img.shape}, wrist image shape: {wrist_img.shape}")

    print(f"  language_instruction: {task_description}")
    log_file.write(f"Episode {episode_idx:03d} ({num_steps} steps): {task_description}\n")

    # Determine success (if last reward is 1)
    done = bool(context["steps/reward"].float_list.value[-1] == 1.0)

    # Save videos
    main_out = os.path.join(output_dir, f"episode_{episode_idx:03d}_main.mp4")
    wrist_out = os.path.join(output_dir, f"episode_{episode_idx:03d}_wrist.mp4")
    save_mp4_video(main_frames, main_out)
    save_mp4_video(wrist_frames, wrist_out)

    # === Save first frame as PNG ===
    Image.fromarray(main_frames[0]).save(os.path.join(output_dir, f"episode_{episode_idx:03d}_first_main_frame.png"))
    Image.fromarray(wrist_frames[0]).save(os.path.join(output_dir, f"episode_{episode_idx:03d}_first_wrist_frame.png"))

log_file.close()
print(f"[Success] Saved log file: {log_file_path}")



# === Aggregate unique language instructions across all shards ===
all_lang_instructions = set()
total_episode_count = 0

print("\n=== Scanning All TFRecord Files for Unique Language Instructions ===")
for i in range(16):
    shard_path = f"/workspace/data/modified_libero_rlds/libero_spatial_no_noops/1.0.0/libero_spatial-train.tfrecord-{i:05d}-of-00016"
    try:
        dataset = tf.data.TFRecordDataset(shard_path)
        for raw_record in dataset:
            example = tf.train.SequenceExample()
            example.ParseFromString(raw_record.numpy())
            context = example.context.feature
            lang = context["steps/language_instruction"].bytes_list.value[0].decode()
            all_lang_instructions.add(lang)
            total_episode_count += 1
    except Exception as e:
        print(f"[Error] Failed to read {shard_path}: {e}")

# === Print Summary ===
print("\n=== Summary ===")
print(f"Total episodes across all shards: {total_episode_count}")
print(f"Unique language instructions found: {len(all_lang_instructions)}")
print("\nUnique Instructions:")
for instr in sorted(all_lang_instructions):
    print(f" - {instr}")