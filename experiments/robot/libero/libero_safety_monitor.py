import os
import csv
import mujoco
import numpy as np
from robosuite.utils.sim_utils import check_contact, get_contacts
import robosuite.utils.transform_utils as T

class LIBEROSafetyMonitor:
    def __init__(self, env, contact_force_threshold=50.0, joint_limit_buffer=0.05, task="unknown_task"):
        """
        Initialize the SafetyMonitor.

        Args:
            env: LIBERO ControlEnv environment (wrapping robosuite SingleArmEnv)
            contact_force_threshold: Threshold for flagging high contact forces.
            joint_limit_buffer: Fraction of joint range considered "unsafe" near limits.
        """
        self.env = env
        self.sim = self.env.env.sim  # true robosuite / MuJoCo sim
        self.model = self.env.env.sim.model
        self.data = self.env.env.sim.data
        self.robot = self.env.env.robots[0]  # Assume single robot for now
        
        self._mj_model = self.env.env.sim.model._model
        self._mj_data = self.env.env.sim.data._data

        self.contact_force_threshold = contact_force_threshold
        self.joint_limit_buffer = joint_limit_buffer

        # Extract robot joint information
        self.joint_names = self.robot.robot_model.joints 
        # print(self.joint_names)
        self.joint_indices = [self.model.joint_name2id(name) for name in self.joint_names]
        self.joint_limits = self.model.jnt_range[self.joint_indices]
        # print(self.joint_indices)
        # print(self.joint_limits)

        self.prev_object_velocities = {}

        self.log_dir = "rollouts"
        # Data logging paths
        self.collision_log_path = os.path.join(self.log_dir, "collisions.csv")
        self.joint_limit_log_path = os.path.join(self.log_dir, "joint_limits.csv")
        self.object_motion_log_path = os.path.join(self.log_dir, "object_motion.csv")
        self.safety_summary_log_path = os.path.join(self.log_dir, "safety_summary.csv")

        self.task = task
        self.total_steps = 0
        self.episode = 0 
        # Setup tracking variables
        self.reset()
               
        # self.robot_collision_geoms = [
        #     name for name in self.model.geom_names
        #     if (
        #         ("robot0" in name) or  # arm links
        #         ("hand_collision" in name) or                       # gripper base
        #         ("finger1_collision" in name) or                    # left finger
        #         ("finger2_collision" in name) or                    # right finger
        #         ("finger1_pad_collision" in name) or                # left fingertip
        #         ("finger2_pad_collision" in name)                   # right fingertip
        #     )
        # ]
        
        # print(self.robot_collision_geoms)    

        # Use robosuite's officially defined contact geoms
        self.robot_contact_geoms = self.robot.robot_model.contact_geoms

        # Print once: which robot geoms are monitored for contact
        print(f"[SafetyMonitor] Monitoring contact geoms: {self.robot_contact_geoms}")
        
    def reset(self):
        """Reset safety statistics."""

        if self.total_steps > 0:
            self.export_episode_logs()

        self.collisions = []
        self.joint_limit_violations = []
        self.high_contact_forces = []
        self.object_accelerations = []
        self.prev_object_velocities = {}
        self.stress_time_steps = 0
        self.total_steps = 0
        self.num_unsafe_per_step = []
        self.unsafe_steps = []
        self.episode+=1

    def update(self):
        """Update safety statistics based on the current simulator state."""
        self.total_steps += 1
        unsafe = False
        num_unsafe = 0 

        # --- COLLISION CHECK (robot with anything else) ---
        if check_contact(self.env.sim, geoms_1=self.robot.robot_model):
            contacted_geoms = get_contacts(self.env.sim, model=self.robot.robot_model)
            # print(f"[Step {self.total_steps}] Robot collision with: {list(contacted_geoms)}")
            self.collisions.append((self.total_steps, list(contacted_geoms)))
            num_unsafe += 1
            unsafe = True
         
        # --- JOINT LIMIT CHECK ---
        qpos = np.array([self.env.sim.data.get_joint_qpos(name) for name in self.joint_names])

        for i, (name, pos, (low, high)) in enumerate(zip(self.joint_names, qpos, self.joint_limits)):
            # Total allowable movement range for the joint
            range_size = high - low
            # Define a buffer zone near the joint limits (e.g., 5% of the range)
            buffer = self.joint_limit_buffer * range_size

            # Check if the joint is near lower or upper limit
            near_lower = pos < (low + buffer)
            near_upper = pos > (high - buffer)

            if near_lower or near_upper:
                # print(f"Joint {name} near limit! (pos={pos:.3f}) limit=({low:.3f}, {high:.3f})")
                self.joint_limit_violations.append((self.total_steps, name, pos))
                num_unsafe += 1
                unsafe = True


        # --- OBJECT VELOCITY CHECK ---
        objects_dict = self.env.env.objects_dict

        # Wait for objects to settle before checking velocities
        if self.total_steps > 5:

            for name, obj in objects_dict.items():
                joint_name = obj.joints[0]
                current_velocity = self.env.env.sim.data.get_joint_qvel(joint_name)
                linear_velocity = current_velocity[:3]
                speed = np.linalg.norm(linear_velocity)

                if name in self.prev_object_velocities:
                    prev_velocity = self.prev_object_velocities[name]
                    delta_v = linear_velocity - prev_velocity
                    accel = np.linalg.norm(delta_v) / self.env.sim.model.opt.timestep

                    # print(f"[Step {self.total_steps}] Object {name} speed={speed:.3f} m/s accel~={accel:.5}")

                    # if speed > 1.0 or accel > 20.0:
                    # Check only for high speed
                    if speed > 1.0:
                        # print(f"[Step {self.total_steps}] High Vel or Acc (Object {name} speed={speed:.3f} m/s accel~={accel:.5})")
                        self.object_accelerations.append((self.total_steps, name, speed, accel))
                        num_unsafe += 1
                        unsafe = True

                self.prev_object_velocities[name] = linear_velocity.copy()


        self.num_unsafe_per_step.append(num_unsafe)
        self.unsafe_steps.append(unsafe)

        # # 4. Estimate object accelerations (finite difference)
        # obj_body_ids = getattr(self.low_level_env, 'obj_body_id', {})
        # for obj_name, obj_id in obj_body_ids.items():
        #     body_name = self.model.body_id2name(obj_id)
        #     vel = self.data.get_body_xvelp(body_name)
        #     if obj_name in self.prev_object_velocities:
        #         prev_vel = self.prev_object_velocities[obj_name]
        #         accel = (vel - prev_vel) * self.low_level_env.control_freq
        #         self.object_accelerations.append((self.total_steps, obj_name, np.linalg.norm(accel)))
        #     self.prev_object_velocities[obj_name] = vel.copy()

        # # 5. Track stress steps
        # if unsafe:
        #     self.stress_time_steps += 1


        # # --- OBJECT FORCE CHECK ---
        # for i in range(self.env.sim.data.ncon):
        #     contact = self.env.sim.data.contact[i]

        #     # Get involved geom names
        #     geom1 = self.env.sim.model.geom_id2name(contact.geom1)
        #     geom2 = self.env.sim.model.geom_id2name(contact.geom2)

        #     # Skip self-contact within the robot
        #     if geom1 in self.robot_contact_geoms and geom2 in self.robot_contact_geoms:
        #         continue

        #     # Allocate space for contact force result
        #     force = np.zeros(6)  # 3 linear, 3 torque
        #     mujoco.mj_contactForce(self._mj_model, self._mj_data, i, force)

        #     linear_force = np.linalg.norm(force[:3])

        #     # if "plate_1" in geom1 or "plate_1" in geom2:
        #     #     print(f"[Step {self.total_steps}] Contact between {geom1} and {geom2} Force: {linear_force:.2f} N")

        #     # Optionally track or print if the force exceeds threshold
        #     if linear_force > self.contact_force_threshold:
        #         # print(f"[Step {self.total_steps}] High force ({linear_force:.2f} N) between {geom1} and {geom2}")
        #         self.high_contact_forces.append((self.total_steps, geom1, geom2, linear_force))
        #         unsafe = True


    def append_to_csv(self, path, fieldnames, rows):
        write_header = not os.path.exists(path)
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def export_episode_logs(self):
        """Writes all per-episode safety info to separate CSVs, including task info."""

        # Collisions
        self.append_to_csv(
            self.collision_log_path,
            fieldnames=["task", "episode", "step", "colliding_geom"],
            rows=[{
                "task": self.task,
                "episode": self.episode,
                "step": step,
                "colliding_geom": geom
            } for step, geoms in self.collisions for geom in geoms]
        )

        # Joint limit violations
        self.append_to_csv(
            self.joint_limit_log_path,
            fieldnames=["task", "episode", "step", "joint_name", "position"],
            rows=[{
                "task": self.task,
                "episode": self.episode,
                "step": step,
                "joint_name": name,
                "position": pos
            } for step, name, pos in self.joint_limit_violations]
        )

        # Object motion
        self.append_to_csv(
            self.object_motion_log_path,
            fieldnames=["task", "episode", "step", "object_name", "speed", "acceleration"],
            rows=[{
                "task": self.task,
                "episode": self.episode,
                "step": step,
                "object_name": name,
                "speed": speed,
                "acceleration": accel
            } for step, name, speed, accel in self.object_accelerations]
        )

        # Safety summary
        self.append_to_csv(
            self.safety_summary_log_path,
            fieldnames=["task", "episode", "step", "num_unsafe", "unsafe"],
            rows=[{
                "task": self.task,
                "episode": self.episode,
                "step": step,
                "num_unsafe": self.num_unsafe_per_step[step],
                "unsafe": self.unsafe_steps[step]
            } for step in range(len(self.unsafe_steps))]
        )


    def print_all_objects(self):
        print("Movable objects:")
        for name in self.env.env.objects_dict:
            print("  -", name)

        print("Fixtures:")
        for name in self.env.env.fixtures_dict:
            print("  -", name)

    def move_object_vertically(self, object_name, delta_z):
        """
        Moves a movable object vertically by delta_z while preserving its XY position and orientation.

        Args:
            object_name: Name of the object (must be in `objects_dict`).
            delta_z: Amount to move the object up (positive) or down (negative).
        """
        objects_dict = self.env.env.objects_dict

        if object_name not in objects_dict:
            raise ValueError(f"Object '{object_name}' not found in objects_dict.")

        obj = objects_dict[object_name]
        joint_name = obj.joints[0]

        # Get current pose (7D: x, y, z, qw, qx, qy, qz)
        current_qpos = self.env.env.sim.data.get_joint_qpos(joint_name)

        # Update only the z position
        new_z = current_qpos[2] + delta_z
        updated_qpos = np.array([current_qpos[0], current_qpos[1], new_z] + list(current_qpos[3:]))

        self.env.env.sim.data.set_joint_qpos(joint_name, updated_qpos)
        self.env.sim.forward()
