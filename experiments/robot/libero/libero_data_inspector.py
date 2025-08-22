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
tfrecord_dir = "/workspace/data/modified_libero_rlds/libero_10_no_noops/1.0.0"

tfrecord_paths = [
    os.path.join(tfrecord_dir, f)
    for f in sorted(os.listdir(tfrecord_dir))
    if ".tfrecord" in f
]
output_dir = "visualize_data"
os.makedirs(output_dir, exist_ok=True)
log_file_path = os.path.join(output_dir, "episode_language_log.txt")
log_file = open(log_file_path, "w")

# === Load all episodes into a list first ===
raw_dataset = list(tf.data.TFRecordDataset(tfrecord_paths))
total_episodes = len(raw_dataset)
print(f"Total episodes in file: {total_episodes}")

# === Sample 10 random episode indices ===
sampled_indices = random.sample(range(total_episodes), min(1, total_episodes))

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


# === Calculate Dataset Statistics ===
# Aggregate unique language instructions across all episodes
all_lang_instructions = set()
total_episode_count = 0

# Track min/max for end effector positions
x_min, x_max = float("inf"), float("-inf")
y_min, y_max = float("inf"), float("-inf")
z_min, z_max = float("inf"), float("-inf")


for raw_record in raw_dataset:
    example = tf.train.SequenceExample()
    example.ParseFromString(raw_record.numpy())
    context = example.context.feature
    lang = context["steps/language_instruction"].bytes_list.value[0].decode()
    all_lang_instructions.add(lang)
    total_episode_count += 1

    # Get trajectory length
    num_steps = len(context["steps/reward"].float_list.value)

    for t in range(num_steps):
        state = context["steps/observation/state"].float_list.value[t*8:(t+1)*8]
        x, y, z = state[0], state[1], state[2]

        # Update min/max
        x_min, x_max = min(x_min, x), max(x_max, x)
        y_min, y_max = min(y_min, y), max(y_max, y)
        z_min, z_max = min(z_min, z), max(z_max, z)

# === Print Summary ===
print("\n=== Summary ===")
print(f"Total episodes across all shards: {total_episode_count}")
print(f"Unique language instructions found: {len(all_lang_instructions)}")
print("\nUnique Instructions:")
for instr in sorted(all_lang_instructions):
    print(f" - {instr}")

# Print EE position bounds
print("\n=== End Effector Position Bounds ===")
print(f"x range: [{x_min:.4f}, {x_max:.4f}]")
print(f"y range: [{y_min:.4f}, {y_max:.4f}]")
print(f"z range: [{z_min:.4f}, {z_max:.4f}]")