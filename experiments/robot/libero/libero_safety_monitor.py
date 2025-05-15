import numpy as np

from robosuite.utils.sim_utils import check_contact, get_contacts

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

        # --- COLLISION CHECK (robot with anything else) ---
        if check_contact(self.env.sim, geoms_1=self.robot.robot_model):
            contacted_geoms = get_contacts(self.env.sim, model=self.robot.robot_model)
            print(f"[Step {self.total_steps}] Robot collision with: {list(contacted_geoms)}")
            self.collisions.append((self.total_steps, list(contacted_geoms)))
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
                print(f"Joint {name} near limit! (pos={pos:.3f}) limit=({low:.3f}, {high:.3f})")
                self.joint_limit_violations.append((self.total_steps, name, pos))
                unsafe = True

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
        #     mujoco.mj_contactForce(self.env.sim.model, self.env.sim.data, i, force)

        #     linear_force = np.linalg.norm(force[:3])

        #     # Optionally track or print if the force exceeds threshold
        #     if linear_force > self.contact_force_threshold:
        #         print(f"[Step {self.total_steps}] High force ({linear_force:.2f} N) between {geom1} and {geom2}")
        #         self.high_contact_forces.append((self.total_steps, geom1, geom2, linear_force))
        #         unsafe = True


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