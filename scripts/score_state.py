"""Classify pen insertion and gripper release from a MuJoCo state."""

from types import SimpleNamespace

import mujoco
import numpy as np

from placement import pen_placement


JOINT_NAMES = ('shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper')
PEN_NAMES = ('pen_0', 'pen_1', 'pen_2')


class StateJudge:
    def __init__(self, scene):
        if mujoco.__version__ != '3.3.7':
            raise RuntimeError('Evaluation requires MuJoCo 3.3.7.')
        model = mujoco.MjModel.from_xml_path(str(scene))
        data = mujoco.MjData(model)
        self.sim = SimpleNamespace(model=model, data=data, holder_id=model.body('pen_holder').id,
                                   pen_ids=[model.body(name).id for name in PEN_NAMES])
        self.joint_addresses = [model.jnt_qposadr[model.joint(name).id] for name in JOINT_NAMES]
        self.pen_addresses = [model.jnt_qposadr[model.joint(name + '_free').id] for name in PEN_NAMES]
        self.jaws = {model.body('gripper').id, model.body('moving_jaw_so101_v1').id}

    def evaluate(self, joint_positions, pen_poses):
        model, data = self.sim.model, self.sim.data
        data.qpos[self.joint_addresses] = joint_positions
        for address, pose in zip(self.pen_addresses, pen_poses):
            data.qpos[address:address + 7] = pose
        mujoco.mj_forward(model, data)
        placed = pen_placement(self.sim)
        touching = np.zeros(3, dtype=bool)
        for contact in data.contact:
            if contact.dist <= 0:
                bodies = {int(model.geom_bodyid[contact.geom1]), int(model.geom_bodyid[contact.geom2])}
                if bodies & self.jaws:
                    for index, pen_id in enumerate(self.sim.pen_ids):
                        touching[index] |= pen_id in bodies
        return placed, placed & ~touching
