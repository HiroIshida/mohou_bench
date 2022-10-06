from dataclasses import dataclass
from typing import Optional, Type

import numpy as np
import pybullet as pb
from skrobot.coordinates import Coordinates

from mohou_bench.robot import (
    CylinderStickPandaModel,
    GripperPandaModel,
    PandaModelBase,
    PybulletRobotInterface,
)


@dataclass
class Commander:
    robot: PandaModelBase
    ri: PybulletRobotInterface
    default_step_length: int = 50
    _init_angle_vector: Optional[np.ndarray] = None

    @classmethod
    def create(cls, robot_type: Type[PandaModelBase] = CylinderStickPandaModel) -> "Commander":
        robot = robot_type()
        ri = PybulletRobotInterface(robot)
        return cls(robot, ri)

    def reset(self) -> None:
        self.ri.reset()
        self.robot.init_pose()

        target = Coordinates(pos=(0.3, -0.1, 0.07))
        target.rotate(np.pi * 0.5, "y")
        target.rotate(np.pi * 0.5, "x")

        if self._init_angle_vector is None:
            self.robot.solve_ik(target)
            av = self.robot.get_joint_angles()
            self._init_angle_vector = av

        self.robot.set_joint_angles(list(self._init_angle_vector))
        self.ri.reset_angles(self.robot)

    def send_command(self, robot_model: PandaModelBase, n_command_split: int = 3):
        # n_command_split to avoid instability due to sudden move of end effector
        joint_angles = robot_model.get_joint_angles()
        if isinstance(robot_model, GripperPandaModel):
            gripper_angles = robot_model.get_gripper_joints()
        else:
            gripper_angles = None

        angles_now = self.ri.get_joint_angles()
        angles_diff = (joint_angles - angles_now) / float(n_command_split)

        if isinstance(robot_model, GripperPandaModel):
            g_now = self.ri.get_gripper_angles()
            g_diff = (gripper_angles - g_now) / float(n_command_split)

        for i in range(n_command_split):
            angles_sub_next = angles_now + angles_diff * (i + 1)
            self.robot.set_joint_angles(angles_sub_next)
            self.ri.reset_angles(self.robot)

            if isinstance(robot_model, GripperPandaModel):
                assert isinstance(self.robot, GripperPandaModel)
                g_sub_next = g_now + g_diff * (i + 1)  # type: ignore
                self.robot.set_gripper_joints(g_sub_next)
                self.ri.reset_gripper_pos(self.robot)

            for _ in range(self.default_step_length):
                pb.stepSimulation()
