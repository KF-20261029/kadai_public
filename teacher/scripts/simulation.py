"""MuJoCo SO-101 observations, joint commands, and physical placement checks."""

from pathlib import Path

import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
JOINT_NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
INITIAL_STATE = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 50.0])


class Simulation:
    def __init__(self, joint_units="degrees", width=640, height=480,
                 servo_damping_ratio=None, interpolate_commands=False, synchronize_observations=True,
                 scene_path=None, render_images=True):
        scene = str(scene_path or ROOT / "simulation/scene.xml")
        if servo_damping_ratio is None:
            self.model = mujoco.MjModel.from_xml_path(scene)
        else:
            if not np.isfinite(servo_damping_ratio) or servo_damping_ratio <= 0:
                raise ValueError("The servo damping ratio must be finite and positive.")
            specification = mujoco.MjSpec.from_file(scene)
            for actuator in specification.actuators:
                actuator.set_to_position(kp=float(actuator.gainprm[0]), dampratio=servo_damping_ratio)
            self.model = specification.compile()
        self.data = mujoco.MjData(self.model)
        self.joint_units = joint_units
        self.interpolate_commands = interpolate_commands
        self.synchronize_observations = synchronize_observations
        self.joint_ids = np.array([self.model.joint(name).id for name in JOINT_NAMES])
        self.qpos_ids = self.model.jnt_qposadr[self.joint_ids]
        self.actuator_ids = np.array([self.model.actuator(name).id for name in JOINT_NAMES])
        self.ranges = self.model.jnt_range[self.joint_ids].copy()
        self.ranges[:, 0] = np.maximum(self.ranges[:, 0], self.model.actuator_ctrlrange[self.actuator_ids, 0])
        self.ranges[:, 1] = np.minimum(self.ranges[:, 1], self.model.actuator_ctrlrange[self.actuator_ids, 1])
        self.model.body_quat[self.model.body("base").id] = [np.sqrt(0.5), 0, 0, -np.sqrt(0.5)]
        for index, color in enumerate(self.model.mat_rgba):
            if np.allclose(color[:3], [1, 0.82, 0.12]):
                self.model.mat_rgba[index] = [0.96, 0.28, 0.06, 1]
        self.pen_ids = np.array([self.model.body(f"pen_{index}").id for index in range(3)])
        self.holder_id = self.model.body("pen_holder").id
        self.renderer = mujoco.Renderer(self.model, height=height, width=width) if render_images else None
        self.width, self.height = width, height

    def to_radians(self, action):
        action = np.asarray(action, dtype=float)
        target = np.empty(6)
        if self.joint_units == "degrees":
            target[:5] = np.deg2rad(action[:5])
        else:
            target[:5] = self.ranges[:5, 0] + (action[:5] + 100) / 200 * np.diff(self.ranges[:5], axis=1)[:, 0]
        target[5] = self.ranges[5, 0] + action[5] / 100 * np.diff(self.ranges[5])[0]
        return np.clip(target, self.ranges[:, 0], self.ranges[:, 1])

    def state(self):
        angles = self.data.qpos[self.qpos_ids]
        values = np.empty(6)
        if self.joint_units == "degrees":
            values[:5] = np.rad2deg(angles[:5])
        else:
            values[:5] = (angles[:5] - self.ranges[:5, 0]) / np.diff(self.ranges[:5], axis=1)[:, 0] * 200 - 100
        values[5] = (angles[5] - self.ranges[5, 0]) / np.diff(self.ranges[5])[0] * 100
        return values.astype(np.float32)

    def reset(self, seed, position_jitter=0.008, yaw_jitter=0.3, initial_state=None):
        if not np.isfinite([position_jitter, yaw_jitter]).all() or min(position_jitter, yaw_jitter) < 0:
            raise ValueError("Initial position and yaw jitter must be finite and non-negative.")
        mujoco.mj_resetData(self.model, self.data)
        rng = np.random.default_rng(seed)
        self.data.qpos[self.qpos_ids] = self.to_radians(INITIAL_STATE if initial_state is None else initial_state)
        self.data.ctrl[self.actuator_ids] = self.data.qpos[self.qpos_ids]
        for index in range(len(self.pen_ids)):
            address = self.model.jnt_qposadr[self.model.joint(f"pen_{index}_free").id]
            self.data.qpos[address:address + 2] += rng.uniform(-position_jitter, position_jitter, 2)
            angle = rng.uniform(-yaw_jitter, yaw_jitter)
            initial_rotation = self.data.qpos[address + 3:address + 7].copy()
            mujoco.mju_mulQuat(self.data.qpos[address + 3:address + 7],
                               np.array([np.cos(angle / 2), 0, 0, np.sin(angle / 2)]), initial_rotation)
        mujoco.mj_forward(self.model, self.data)
        for _ in range(90):
            mujoco.mj_step(self.model, self.data)
        if self.synchronize_observations:
            mujoco.mj_forward(self.model, self.data)
        self.data.time = 0
        self.initial_pen_positions = self.pen_positions()
        if self.placement().any():
            raise RuntimeError("An object started inside the goal; this would invalidate the evaluation.")

    def step(self, action):
        target = self.to_radians(action)
        previous_target = self.data.ctrl[self.actuator_ids].copy()
        for substep in range(10):
            self.data.ctrl[self.actuator_ids] = (
                previous_target + (target - previous_target) * (substep + 1) / 10
                if self.interpolate_commands else target
            )
            mujoco.mj_step(self.model, self.data)
        if self.synchronize_observations:
            mujoco.mj_forward(self.model, self.data)
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise RuntimeError("The physical simulation became non-finite.")
        return target

    def render(self, camera):
        if self.renderer is None:
            raise RuntimeError("Rendering is disabled for this physics-only simulation.")
        self.renderer.update_scene(self.data, camera=camera)
        return self.renderer.render().copy()

    def observation(self, instruction):
        if self.synchronize_observations:
            mujoco.mj_forward(self.model, self.data)
        return {
            "observation.images.top": torch.from_numpy(self.render("top")).permute(2, 0, 1).float() / 255,
            "observation.images.wrist": torch.from_numpy(self.render("wrist_cam")).permute(2, 0, 1).float() / 255,
            "observation.state": torch.from_numpy(self.state()),
            "task": instruction,
        }

    def pen_positions(self):
        return self.data.xpos[self.pen_ids].copy()

    def placement(self):
        positions = self.pen_positions()
        axes = self.data.xmat[self.pen_ids].reshape(-1, 3, 3)[:, :, 0]
        lower_end = positions - axes * (np.sign(axes[:, 2]) * 0.065)[:, None]
        holder = self.data.xpos[self.holder_id]
        radial = np.linalg.norm(lower_end[:, :2] - holder[:2], axis=1)
        return (
            (radial < 0.028)
            & (lower_end[:, 2] > 0.004)
            & (lower_end[:, 2] < 0.022)
            & (positions[:, 2] > 0.04)
            & (np.abs(axes[:, 2]) > 0.65)
        )

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
