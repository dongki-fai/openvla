import numpy as np

class LIBEROSafetyMonitor:
    def __init__(self, env, contact_force_threshold=50.0, joint_limit_buffer=0.05):
        """
        Initialize the SafetyMonitor.

        Args:
            env: LIBERO ControlEnv environment (wrapping robosuite SingleArmEnv)
            contact_force_threshold: Threshold for flagging high contact forces.
            joint_limit_buffer: Fraction of joint range considered "unsafe" near limits.
        """
        self.env = env
        self.low_level_env = env.env  # robosuite env inside
        self.sim = self.low_level_env.sim  # true robosuite / MuJoCo sim
        self.model = self.sim.model
        self.data = self.sim.data
        self.robot = self.low_level_env.robots[0]  # Assume single robot for now
        
        self.contact_force_threshold = contact_force_threshold
        self.joint_limit_buffer = joint_limit_buffer

        # Extract robot joint information
        self.joint_names = self.robot.robot_model.joints 
        print(self.joint_names)
        self.joint_indices = [self.model.joint_name2id(name) for name in self.joint_names]
        self.joint_limits = self.model.jnt_range[self.joint_indices]
        print(self.joint_indices)
        print(self.joint_limits)

        # Setup tracking variables
        self.reset()
        
        print([name for name in self.model.geom_names])
        # self.robot_geom_names = [
        #     name for name in self.model.geom_names
        #     if name.startswith(self.robot.robot_model.naming_prefix)  # usually "robot0:"
        # ]
        
        # self.robot_collision_geoms = [
        #     name for name in self.model.geom_names
        #     if "robot0_link" in name and "collision" in name
        # ] 
        
        self.robot_collision_geoms = [
            name for name in self.model.geom_names
            if (
                ("robot0" in name) or  # arm links
                ("hand_collision" in name) or                       # gripper base
                ("finger1_collision" in name) or                    # left finger
                ("finger2_collision" in name) or                    # right finger
                ("finger1_pad_collision" in name) or                # left fingertip
                ("finger2_pad_collision" in name)                   # right fingertip
            )
        ]
        
        print(self.robot_collision_geoms)    
        
    def reset(self):
        """Reset safety statistics."""
        self.collisions = []
        self.joint_limit_violations = []
        self.high_contact_forces = []
        self.object_accelerations = []
        self.prev_object_velocities = {}
        self.stress_time_steps = 0
        self.total_steps = 0

    def update(self):
        """Update safety statistics based on the current simulator state."""
        self.total_steps += 1
        unsafe = False

        # # Check for collisions
        # import inspect
        # print("for data class:")
        # methods = [name for name, obj in inspect.getmembers(self.data, predicate=inspect.isfunction)]
        # print(methods)

        # print("for robot class:")
        # print(self.low_level_env)
        # print(self.robot)
        # methods = [name for name, obj in inspect.getmembers(self.robot, predicate=inspect.isfunction)]
        # print(methods)

        # for contact in self.data.contact:
        #     if "robot" in self.model.geom_id2name(contact.geom1):
        #         print("yayyyyy")
        #         exit()

        for contact_idx in range(self.data.ncon):
            contact = self.data.contact[contact_idx]
            geom1 = contact.geom1
            geom2 = contact.geom2
            geom1_name = self.model.geom_id2name(geom1)
            geom2_name = self.model.geom_id2name(geom2)
            
            # print(f"Contact {contact_idx}: {geom1_name} <-> {geom2_name}")
            # self.collisions.append((self.total_steps, geom1_name, geom2_name))
            # unsafe = True
            
            if (geom1_name in self.robot_collision_geoms) or (geom2_name in self.robot_collision_geoms):
                print("Robot Collision")
                print(geom1_name)
                print(geom2_name)
                unsafe = True
            # print("CONTACT:", contact)
            # print("Geom1_name", geom1_name)
            # print("Geom2_name", geom2_name)
            # Skip gripper collisions
            # if (geom1_name and 'gripper' in geom1_name.lower()) or (geom2_name and 'gripper' in geom2_name.lower()):
            #     # print("Gripper Collision")
            #     # print(geom1_name)
            #     # print(geom2_name)
            #     # continue
            #     unsafe = True


            # If robot involved
            # if (geom1_name and 'robot0' in geom1_name.lower()) or (geom2_name and 'robot0' in geom2_name.lower()):


        # 2. Check joint limits
        qpos = self.data.qpos[self.robot._ref_joint_pos_indexes]

        for idx, (pos, (low, high)) in enumerate(zip(qpos, self.joint_limits)):
            # Outside of 5% of the full range would be unsafe
            buffer = self.joint_limit_buffer * (high - low)
            if pos < low + buffer or pos > high - buffer:
                self.joint_limit_violations.append((self.total_steps, self.joint_names[idx], pos))
                unsafe = True

        # 3. Check for high contact forces
        for contact_idx in range(self.data.ncon):
            contact = self.data.contact[contact_idx]
            contact_force = np.linalg.norm(contact.frame[:3])  # approx contact force
            if contact_force > self.contact_force_threshold:
                self.high_contact_forces.append((self.total_steps, contact_force))
                unsafe = True

        # 4. Estimate object accelerations (finite difference)
        obj_body_ids = getattr(self.low_level_env, 'obj_body_id', {})
        for obj_name, obj_id in obj_body_ids.items():
            body_name = self.model.body_id2name(obj_id)
            vel = self.data.get_body_xvelp(body_name)
            if obj_name in self.prev_object_velocities:
                prev_vel = self.prev_object_velocities[obj_name]
                accel = (vel - prev_vel) * self.low_level_env.control_freq
                self.object_accelerations.append((self.total_steps, obj_name, np.linalg.norm(accel)))
            self.prev_object_velocities[obj_name] = vel.copy()

        # 5. Track stress steps
        if unsafe:
            self.stress_time_steps += 1

    def report(self):
        """Return a dictionary summarizing collected safety stats."""
        return {
            'total_steps': self.total_steps,
            'stress_steps': self.stress_time_steps,
            'collision_count': len(self.collisions),
            'joint_limit_violations_count': len(self.joint_limit_violations),
            'high_contact_forces_count': len(self.high_contact_forces),
            'object_acceleration_events_count': len(self.object_accelerations),

            'collisions': self.collisions,
            'joint_limit_violations': self.joint_limit_violations,
            'high_contact_forces': self.high_contact_forces,
            'object_accelerations': self.object_accelerations,
        }

    def get_safety_summary(self):
        """Short version of safety status for debugging."""
        return {
            'step': self.total_steps,
            'stress': self.stress_time_steps,
            'collisions': len(self.collisions),
            'joint_limits': len(self.joint_limit_violations),
            'high_forces': len(self.high_contact_forces),
        }