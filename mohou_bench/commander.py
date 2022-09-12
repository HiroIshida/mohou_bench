from typing import Optional

import numpy as np
import pybullet as pb
from skrobot.coordinates import Coordinates

from mohou_bench.robot import PybulletRobotInterface, StickPandaModel


class Commander:
    robot: StickPandaModel
    ri: PybulletRobotInterface
    default_step_length: int = 50
    _init_angle_vector: Optional[np.ndarray] = None

    def __init__(self, robot: StickPandaModel, ri: PybulletRobotInterface):
        self.robot = robot
        self.ri = ri

    @classmethod
    def create(cls) -> "Commander":
        robot = StickPandaModel()
        ri = PybulletRobotInterface(robot)
        return cls(robot, ri)

    def reset(self) -> None:
        self.ri.reset()
        self.robot.init_pose()

        target = Coordinates(pos=(0.3, 0.0, 0.07))
        target.rotate(np.pi * 0.5, "y")
        target.rotate(np.pi * 0.5, "x")

        if self._init_angle_vector is None:
            self.robot.solve_ik(target)
            av = self.robot.get_joint_angles()
            self._init_angle_vector = av

        self.robot.set_joint_angles(list(self._init_angle_vector))
        self.ri.reset_angles(self.robot)

    def send_command(self, joint_angles: np.ndarray):
        n_command_split = 3  # to avoid instability due to sudden move of end effector
        angles_now = self.ri.get_joint_angles()
        angles_diff = (joint_angles - angles_now) / float(n_command_split)

        for i in range(n_command_split):
            angles_sub_next = angles_now + angles_diff * (i + 1)
            self.robot.set_joint_angles(angles_sub_next)
            self.ri.reset_angles(self.robot)
            for _ in range(self.default_step_length):
                pb.stepSimulation()
