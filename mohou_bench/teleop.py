import threading
import time
from datetime import datetime
from enum import Enum
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import pybullet as pb
import pygame
from pynput.keyboard import Key
from skrobot.coordinates import Coordinates

from mohou_bench.robot import IKFailError, PybulletRobotInterface, StickPandaModel


class PS4Button(Enum):
    C = ROSS = 0
    CIRCLE = 1
    TRIANGLE = 2
    SQUARE = 3
    L1 = 4
    L2 = 5


class PS4ControllerManager(threading.Thread):
    controller: pygame.joystick.Joystick
    button_values: List[Optional[bool]]
    joy_vector: Optional[np.ndarray] = None
    is_running: bool = True

    def __init__(self):
        pygame.init()
        pygame.joystick.init()
        count = pygame.joystick.get_count()
        assert count == 1
        controller = pygame.joystick.Joystick(0)
        controller.init()
        self.controller = controller
        self.button_values = [None for _ in range(len(PS4Button))]

        super().__init__()

    def run(self):
        while self.is_running:
            for e in pygame.event.get():
                if e.type == pygame.JOYAXISMOTION:
                    vector = np.array([self.controller.get_axis(0), self.controller.get_axis(1)])
                    vector = np.flip(vector)
                    self.joy_vector = vector * 0.005

                if e.type in [pygame.JOYBUTTONDOWN, pygame.JOYBUTTONUP]:
                    for i in range(len(PS4Button)):
                        self.button_values[i] = bool(self.controller.get_button(i))

                if e.type == pygame.JOYBUTTONDOWN:
                    if self.button_values[PS4Button.L1.value]:
                        self.is_running = False


class TeleoperationCommander:
    robot: StickPandaModel
    ri: PybulletRobotInterface
    press_time_table: Dict[Key, datetime]
    post_command_hook: Optional[Callable] = None
    freq: float = 0.01
    delta: float = 0.05
    default_step_length: int = 30

    def __init__(self, robot: StickPandaModel, ri: PybulletRobotInterface):
        self.robot = robot
        self.ri = ri
        self.press_time_table = {}

    @classmethod
    def create(cls) -> "TeleoperationCommander":
        robot = StickPandaModel()
        ri = PybulletRobotInterface(robot)

        robot.init_pose()
        target = Coordinates(pos=(0.3, 0.0, 0.07))
        target.rotate(np.pi * 0.5, "y")
        target.rotate(np.pi * 0.5, "x")
        robot.solve_ik(target)
        ri.reset_angles(robot)
        return cls(robot, ri)

    def on_press(self, key: Key):
        self.press_time_table[key] = datetime.now()

    def process_command(self) -> Tuple[Set, np.ndarray]:
        now = datetime.now()

        activated_keys = set()
        for key, val in self.press_time_table.items():
            delta = (now - val).total_seconds()
            if delta < self.freq:
                if isinstance(key, Key):
                    activated_keys.add(key)
                else:
                    activated_keys.add(key.char)

        command_2d = np.zeros(2)
        if Key.right in activated_keys:
            command_2d[0] += self.delta
        if Key.left in activated_keys:
            command_2d[0] -= self.delta
        if Key.up in activated_keys:
            command_2d[1] += self.delta
        if Key.down in activated_keys:
            command_2d[1] -= self.delta
        return activated_keys, command_2d

    def run(self):
        # listener = Listener(on_press=self.on_press)
        # listener.start()
        ps4_manager = PS4ControllerManager()
        ps4_manager.start()

        while True:
            time.sleep(self.freq)
            # activated_keys, command_2d = self.process_command()
            # if "q" in activated_keys:
            #    break
            if not ps4_manager.is_running:
                break
            command_2d = ps4_manager.joy_vector
            if command_2d is not None and np.linalg.norm(command_2d) > 0:
                command_3d = np.array([command_2d[0], command_2d[1], 0.0])
                try:
                    self.robot.move_end_pos(command_3d, wrt="world")
                    self.robot.get_end_effector()
                    self.ri.reset_angles(self.robot)
                    if self.post_command_hook is not None:
                        self.post_command_hook(self)
                except IKFailError:
                    print("IK fail!")

            for _ in range(self.default_step_length):
                pb.stepSimulation()
