from libero.libero.envs.env_wrapper import ControlEnv
from experiments.robot.libero.libero_safety_monitor import LIBEROSafetyMonitor
from libero.libero import benchmark
from experiments.robot.libero.libero_utils import get_libero_env, get_libero_dummy_action, save_rollout_video
import numpy as np 


USE_LIVE_VIEWER = True  # Set to False for headless mode


# Load LIBERO task suite
benchmark_dict = benchmark.get_benchmark_dict()
task_suite = benchmark_dict["libero_spatial"]()  # or libero_object, etc.
task = task_suite.get_task(0)

# Create LIBERO environment
env, task_description = get_libero_env(task, 
                                       model_family="openvla",     
                                       resolution=256, 
                                       use_viewer=USE_LIVE_VIEWER)


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
    
    if not USE_LIVE_VIEWER:
        # Save agent view image
        img = obs["agentview_image"]
        img = img[::-1, ::-1]  # Rotate 180 degrees (if your libero_utils expects that)
        replay_images.append(img)

    # Dummy action (could be openvla model output)
    # action = get_libero_dummy_act* ion(model_family="op* envla")* * * 
    
    action_dim = env.env.robots[0].action_dim
    action = np.zeros(action_dim)
    action[1] = -1 * np.sin(10 * step_idx)  # Shoulder down
    action[2] = -10.8 * np.sin(5 * step_idx) # Elbow down
    action[0] = -10.2 * np.sin(0.7 * step_idx)  # Base sweep
    action[4] = -4.6 * np.sin(0.07 * step_idx)  # Wrist motion
    action[5] = -0.2 * np.sin(0.8 * step_idx)  
    action[3] = -0.6 * np.sin(1.5 * step_idx)  
    noise = np.random.normal(scale=5, size=action_dim)
    action += noise

    # Step environment
    obs, reward, done, info = env.step(action)# Render in interactive viewer (if enabled)
    
    if USE_LIVE_VIEWER:
        env.env.render() 
    
    # Update safety monitor
    safety_monitor.update()

    # Print current safety info
    # print(f"Step {step_idx} Safety Info: {safety_monitor.get_safety_summary()}")

    if done:
        break

# Close environment
env.close()

if not USE_LIVE_VIEWER:
    # Save replay video
    save_rollout_video(replay_images, idx=0, success=done, task_description=task_description)