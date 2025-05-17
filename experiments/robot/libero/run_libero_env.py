from libero.libero.envs.env_wrapper import ControlEnv
from experiments.robot.libero.libero_safety_monitor import LIBEROSafetyMonitor
from libero.libero import benchmark
from experiments.robot.libero.libero_utils import get_libero_env, get_libero_dummy_action, save_rollout_video
from experiments.robot.libero.libero_safety_monitor_test import LIBEROSafetyMonitorTest
import numpy as np 
# from robosuite.renderers.viewer.mjviewer_renderer import MjviewerRenderer
from mujoco import viewer as mj_viewer
import mujoco


USE_LIVE_VIEWER = False  # Live Viewer
USE_INTERACTIVE_VIEWER = False  # Interactive view
USE_VIEWER = USE_LIVE_VIEWER or USE_INTERACTIVE_VIEWER
TEST_SAFETY_MONITOR = False  # Test safety monitor

# Load LIBERO task suite
benchmark_dict = benchmark.get_benchmark_dict()
task_suite = benchmark_dict["libero_spatial"]()  # or libero_object, etc.
task = task_suite.get_task(0)

# Create LIBERO environment
env, task_description = get_libero_env(task, 
                                       model_family="openvla",     
                                       resolution=256, 
                                       use_viewer=USE_VIEWER)

if USE_INTERACTIVE_VIEWER:
    sim = env.env.sim
    viewer = mj_viewer.launch_passive(sim.model._model, sim.data._data)

    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True

    # Optional: adjust scale of contact force arrows
    sim.model.vis.scale.contactwidth = 0.01
    sim.model.vis.scale.contactheight = 0.01

# Initialize safety monitor
safety_monitor = LIBEROSafetyMonitor(env)

if TEST_SAFETY_MONITOR:
    safety_monitor_test = LIBEROSafetyMonitorTest(env)

# # Recompute contacts and constraint forces
# mujoco.mj_forward(env.env.sim.model._model, env.env.sim.data._data)


# if TEST_SAFETY_MONITOR:
#     safety_monitor_test.lift_object_above_table("cookies_1_g1", target_z=10)


# safety_monitor.move_object_vertically("plate_1", delta_z=.75)


# Dummy policy rollout
num_steps_per_episode = 100
num_episodes = 5


for episode_idx in range(num_episodes):
    # Reset environment and safety monitor
    obs = env.reset()
    safety_monitor.reset()

    # Store images for video
    replay_images = []

    for step_idx in range(num_steps_per_episode):
        
        if not USE_LIVE_VIEWER:
            # Save agent view image
            img = obs["agentview_image"]
            img = img[::-1, ::-1]  # Rotate 180 degrees (if your libero_utils expects that)
            replay_images.append(img)

        # Dummy action (could be openvla model output)
        # action = get_libero_dummy_act* ion(model_family="op* envla")* * * 
        

        # [[-2.897 2.897]
        # [-1.763 1.763]
        # [-2.897 2.897]
        # [-3.072 -0.070]
        # [-2.897 2.897]
        # [-0.018 3.752]
        # [-2.897 2.897]]

        action_dim = env.env.robots[0].action_dim
        action = np.zeros(action_dim)
        action[0] = 0 
        action[1] = 1.7
        action[2] = -2.87
        action[3] = -3
        # noise = np.random.normal(scale=5, size=action_dim)
        # action += noise

        # Step environment
        obs, reward, done, info = env.step(action)# Render in interactive viewer (if enabled)
        
        if USE_LIVE_VIEWER:
            env.env.render() 
        if USE_INTERACTIVE_VIEWER:
            viewer.sync() 
            
        # Update safety monitor
        safety_monitor.update()

        # mujoco.mj_forward(env.env.sim.model._model, env.env.sim.data._data)

        # Print current safety info
        # print(f"Step {step_idx} Safety Info: {safety_monitor.get_safety_summary()}")

        if done:
            break

    if not USE_LIVE_VIEWER:
        # Save replay video
        save_rollout_video(replay_images, idx=0, success=done, task_description=task_description)

# Close environment
env.close()

