from experiments.robot.libero.libero_safety_monitor import LIBEROSafetyMonitor
import mujoco
import numpy as np

class LIBEROSafetyMonitorTest(LIBEROSafetyMonitor):
    def lift_object_above_table(self, object_geom_name, target_z=1.0, delta_z=0.0):
        """
        Lifts the MuJoCo body associated with a named geom (e.g. 'glazed_rim_..._g1') to a height.

        Args:
            object_geom_name (str): Name of one of the geoms on the object.
            target_z (float): Desired z-height.
            delta_z (float): Offset from current z-position.

        Raises:
            ValueError: If no movable body is found.
        """


        print("See the types")
        model = self.sim.model._model
        data = self.sim.data._data

        # model = self.sim.model
        # data = self.sim.data
        

        # model = self._mj_model
        # data = self._mj_data

        # Get geom and owning body
        try:
            geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, object_geom_name)
        except mujoco.MujocoException:
            raise ValueError(f"Geom '{object_geom_name}' not found.")

        body_id = model.geom_bodyid[geom_id]
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)

        # Confirm it has a free joint
        joint_adr = model.body_jntadr[body_id]
        joint_type = model.jnt_type[joint_adr]

        if joint_type != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError(f"Body '{body_name}' is not a free body and cannot be repositioned.")

        # Get qpos addr
        qposadr = model.jnt_qposadr[joint_adr]

        # Modify position and orientation
        current_pos = np.array(data.qpos[qposadr:qposadr+3])
        new_pos = current_pos.copy()

        if delta_z != 0.0:
            new_pos[2] += delta_z
        else:
            new_pos[2] = target_z

        # Use current orientation (quaternion)
        quat = data.qpos[qposadr+3:qposadr+7]

        data.qpos[qposadr:qposadr+3] = new_pos
        data.qpos[qposadr+3:qposadr+7] = quat
        data.qvel[qposadr:qposadr+6] = 0

        mujoco.mj_forward(model, data)

        print(f"[SafetyTest] Lifted body '{body_name}' to z={new_pos[2]:.2f}")