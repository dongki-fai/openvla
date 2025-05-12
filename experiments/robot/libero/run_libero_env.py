# from libero.libero import benchmark
# from experiments.robot.libero.libero_utils import get_libero_env, get_libero_dummy_action, save_rollout_video

# # Load LIBERO task suite
# benchmark_dict = benchmark.get_benchmark_dict()
# task_suite = benchmark_dict["libero_spatial"]()  # or libero_object, etc.
# task = task_suite.get_task(0)

# # Create LIBERO environment
# env, task_description = get_libero_env(task, model_family="openvla", resolution=256)

# # Reset environment
# env.reset()

# # Dummy policy interaction
# replay_images = []
# obs = env.reset()
# num_steps = 300

# for t in range(num_steps):
#     # Get image
#     img = obs["agentview_image"]
#     img = img[::-1, ::-1]  # Rotate 180 degrees like libero_utils expects
#     replay_images.append(img)

#     # Dummy no-op action
#     action = get_libero_dummy_action(model_family="openvla")
#     obs, reward, done, info = env.step(action)

#     if done:
#         break

# # Save replay video
# save_rollout_video(replay_images, idx=0, success=done, task_description=task_description)


from libero.libero.envs.env_wrapper import ControlEnv
from experiments.robot.libero.libero_safety_monitor import LIBEROSafetyMonitor
from libero.libero import benchmark
from experiments.robot.libero.libero_utils import get_libero_env, get_libero_dummy_action, save_rollout_video

import numpy as np 

# Load LIBERO task suite
benchmark_dict = benchmark.get_benchmark_dict()
task_suite = benchmark_dict["libero_spatial"]()  # or libero_object, etc.
task = task_suite.get_task(0)

# Create LIBERO environment
env, task_description = get_libero_env(task, model_family="openvla", resolution=256)

# Initialize safety monitor
safety_monitor = LIBEROSafetyMonitor(env)

# Reset environment and safety monitor
obs = env.reset()
safety_monitor.reset()

# Store images for video
replay_images = []

# Dummy policy rollout
num_steps = 300
for step_idx in range(num_steps):
    # Save agent view image
    img = obs["agentview_image"]
    img = img[::-1, ::-1]  # Rotate 180 degrees (if your libero_utils expects that)
    replay_images.append(img)

    # Dummy action (could be openvla model output)
    # action = get_libero_dummy_action(model_family="openvla")
    
    action_dim = env.env.robots[0].action_dim  # Access action_dim from robot directly
    base_action = 0.7 * np.sin(0.15 * step_idx) * np.ones(action_dim)
    noise = np.random.normal(scale=0.2, size=action_dim)
    action = np.clip(base_action + noise, -1.0, 1.0)

    # action_dim = env.env.robots[0].action_dim  # Access robot's action dimension

    # # Base: Combination of different sine waves with random frequencies and phases
    # frequencies = np.random.uniform(0.02, 0.1, size=action_dim)
    # phases = np.random.uniform(0, 2*np.pi, size=action_dim)
    # base_action = 0.5 * np.sin(frequencies * step_idx + phases)

    # # Add some random drift + noise
    # drift = 0.1 * np.tanh(0.001 * step_idx) * np.random.uniform(-1, 1, size=action_dim)
    # noise = np.random.normal(scale=0.2, size=action_dim)

    # # Final action
    # action = np.clip(base_action + drift + noise, -1.0, 1.0)

    # Step environment
    obs, reward, done, info = env.step(action)

    # Update safety monitor
    safety_monitor.update()

    # Print current safety info
    print(f"Step {step_idx} Safety Info: {safety_monitor.get_safety_summary()}")

    if done:
        break

# Close environment
env.close()

# Save replay video
save_rollout_video(replay_images, idx=0, success=done, task_description=task_description)