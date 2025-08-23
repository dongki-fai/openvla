from libero.libero.envs.env_wrapper import ControlEnv
from experiments.robot.libero.libero_safety_monitor import LIBEROSafetyMonitor
from libero.libero import benchmark
from experiments.robot.libero.libero_utils import get_libero_env, get_libero_dummy_action, save_rollout_video
from experiments.robot.libero.libero_safety_monitor_test import LIBEROSafetyMonitorTest
import numpy as np 
# from robosuite.renderers.viewer.mjviewer_renderer import MjviewerRenderer
from mujoco import viewer as mj_viewer
import mujoco
import os 
import tensorflow as tf

USE_LIVE_VIEWER = False  # Live Viewer
USE_INTERACTIVE_VIEWER = False  # Interactive view
USE_VIEWER = USE_LIVE_VIEWER or USE_INTERACTIVE_VIEWER
TEST_SAFETY_MONITOR = False  # Test safety monitor
USE_ACTIONS_FROM_DATASET = True  # Use actions from dataset (if False, use dummy actions)
LIBERO_BENCHMARK = "libero_spatial"  # libero_spatial, libero_object, etc.

# Load LIBERO task suite
benchmark_dict = benchmark.get_benchmark_dict()
task_suite = benchmark_dict[LIBERO_BENCHMARK]()
task = task_suite.get_task(0)
task_name_to_id_dict = {task_suite.get_task(t).name: t for t in range(task_suite.get_num_tasks())}

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

if USE_ACTIONS_FROM_DATASET:
    tfrecord_dir = f"/workspace/data/modified_libero_rlds/{LIBERO_BENCHMARK}_no_noops/1.0.0"

    tfrecord_paths = [
        os.path.join(tfrecord_dir, f)
        for f in sorted(os.listdir(tfrecord_dir))
        if ".tfrecord" in f
    ]

    raw_dataset = list(tf.data.TFRecordDataset(tfrecord_paths))
    total_episodes = len(raw_dataset)
    print(f"Total episodes in file: {total_episodes}")
else:
    # Dummy policy rollout
    num_steps_per_episode = 60
    total_episodes = 3

# Initialize safety monitor
safety_monitor = LIBEROSafetyMonitor(env)

if TEST_SAFETY_MONITOR:
    safety_monitor_test = LIBEROSafetyMonitorTest(env)

# # Recompute contacts and constraint forces
# mujoco.mj_forward(env.env.sim.model._model, env.env.sim.data._data)


# if TEST_SAFETY_MONITOR:
#     safety_monitor_test.lift_object_above_table("cookies_1_g1", target_z=10)

for episode_idx in range(total_episodes):

    print("===== Starting episode:", episode_idx)
    # Reset environment and safety monitor
    obs = env.reset()
    safety_monitor.reset()

    # safety_monitor.move_object_vertically("plate_1", delta_z=.75)

    if USE_ACTIONS_FROM_DATASET:
        raw_record = raw_dataset[episode_idx]
        example = tf.train.SequenceExample()
        example.ParseFromString(raw_record.numpy())
        context = example.context.feature
        num_steps_per_episode = len(context["steps/reward"].float_list.value)
        
        file_path = context["episode_metadata/file_path"].bytes_list.value[0].decode()
        basename = os.path.basename(file_path) 
        task_name = basename.replace("_demo.hdf5", "")
        task_id = task_name_to_id_dict[task_name]

        # Get task
        task = task_suite.get_task(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = get_libero_env(task,
                                                'openvla', 
                                                resolution=256, 
                                                use_viewer=USE_VIEWER)

        # Initialize safety monitor
        safety_monitor = LIBEROSafetyMonitor(env, task=task.name)
    
    # Store images for video
    replay_images = []

    for step_idx in range(num_steps_per_episode):
        
        if not USE_LIVE_VIEWER:
            # Save agent view image
            img = obs["agentview_image"]
            img = img[::-1, ::-1]  # Rotate 180 degrees (if your libero_utils expects that)
            replay_images.append(img)

        if USE_ACTIONS_FROM_DATASET:
            action = context["steps/action"].float_list.value[step_idx*7:(step_idx+1)*7]
            action = np.array(action)
        else:
            # Dummy action (could be openvla model output)
            # action = get_libero_dummy_action(model_family="openvla")
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
        save_rollout_video(replay_images, idx=episode_idx, success=done, task_description=task_description)

# Close environment
env.close()

