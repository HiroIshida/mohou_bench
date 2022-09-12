import time
from datetime import datetime
from typing import Dict

import numpy as np
import pybullet as pb
from pynput.keyboard import Key, Listener
from skrobot.coordinates import Coordinates

from mohou_bench.robot import PybulletRobotInterface, StickPandaModel


class KeyboardCommander:
    robot: StickPandaModel
    ri: PybulletRobotInterface
    press_time_table: Dict[Key, datetime]
    freq: float = 0.1
    delta: float = 0.003

    def __init__(self, robot: StickPandaModel, ri: PybulletRobotInterface):
        self.robot = robot
        self.ri = ri
        self.press_time_table = {}

    @classmethod
    def create(cls) -> "KeyboardCommander":
        robot = StickPandaModel()
        ri = PybulletRobotInterface(robot)

        robot.init_pose()
        target = Coordinates(pos=(0.4, 0.0, 0.3))
        target.rotate(np.pi * 0.5, "y")
        target.rotate(np.pi * 0.5, "x")
        robot.solve_ik(target)
        ri.reset_angles(robot)
        return cls(robot, ri)

    def on_press(self, key: Key):
        self.press_time_table[key] = datetime.now()

    def get_2d_command(self) -> np.ndarray:
        now = datetime.now()

        activated_keys = set()
        for key, val in self.press_time_table.items():
            delta = (now - val).total_seconds()
            if delta < self.freq:
                activated_keys.add(key)

        command_2d = np.zeros(2)
        if Key.right in activated_keys:
            command_2d[0] += self.delta
        if Key.left in activated_keys:
            command_2d[0] -= self.delta
        if Key.up in activated_keys:
            command_2d[1] += self.delta
        if Key.down in activated_keys:
            command_2d[1] -= self.delta
        return command_2d

    def run(self):
        listener = Listener(on_press=self.on_press)
        listener.start()

        while True:
            time.sleep(self.freq)
            command_2d = self.get_2d_command()
            if np.linalg.norm(command_2d) > 0:
                command_3d = np.array([command_2d[0], command_2d[1], 0.0])
                self.robot.move_end_pos(command_3d, wrt="world")
                self.robot.get_end_effector()
                self.ri.reset_angles(self.robot)
            pb.stepSimulation()


if __name__ == "__main__":
    pb.connect(pb.GUI)
    com = KeyboardCommander.create()
    com.run()
