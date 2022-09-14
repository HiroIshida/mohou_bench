import threading
import time
from datetime import datetime
from enum import Enum
from typing import Callable, Dict, Optional, Set, Tuple, Type

import numpy as np
import pygame
from pynput.keyboard import Key

from mohou_bench.commander import Commander
from mohou_bench.robot import CylinderStickPandaModel, IKFailError, StickPandaModelBase


class PS4Button(Enum):
    CROSS = 0
    CIRCLE = 1
    TRIANGLE = 2
    SQUARE = 3
    L1 = 4
    R1 = 5


class PS4ControllerManager(threading.Thread):
    controller: pygame.joystick.Joystick
    button_values: Dict[PS4Button, Optional[bool]]
    button_down_callbacks: Dict[PS4Button, Callable]
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
        self.button_down_callbacks = {}
        self.button_values = {b: None for b in PS4Button}
        self.register_callback(PS4Button.L1, self.finish)

        super().__init__()

    def register_callback(self, button: PS4Button, func: Callable):
        assert button not in self.button_down_callbacks, "no override is accepted"
        self.button_down_callbacks[button] = func

    def finish(self):
        self.is_running = False

    def run(self):
        while self.is_running:
            time.sleep(1e-4)
            for e in pygame.event.get():
                if e.type == pygame.JOYAXISMOTION:
                    vector = np.array([self.controller.get_axis(0), self.controller.get_axis(1)])
                    vector = np.flip(vector)
                    self.joy_vector = vector * 0.01

                if e.type in [pygame.JOYBUTTONDOWN, pygame.JOYBUTTONUP]:
                    for button in PS4Button:
                        idx = button.value
                        self.button_values[button] = bool(self.controller.get_button(idx))

                if e.type == pygame.JOYBUTTONDOWN:
                    for button in self.button_down_callbacks.keys():
                        if self.button_values[button]:
                            self.button_down_callbacks[button]()


class TeleoperationCommander:
    commander: Commander
    press_time_table: Dict[Key, datetime]
    ps4_manager: PS4ControllerManager
    post_command_hook: Optional[Callable] = None
    delta: float = 0.05
    freq: float = 0.01
    default_step_length: int = 50

    def __init__(self, commander: Commander):
        self.commander = commander
        self.press_time_table = {}
        self.ps4_manager = PS4ControllerManager()
        self.commander.reset()

    @classmethod
    def create(
        cls, robot_type: Type[StickPandaModelBase] = CylinderStickPandaModel
    ) -> "TeleoperationCommander":
        return cls(Commander.create(robot_type))

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

    @property
    def robot(self) -> StickPandaModelBase:
        return self.commander.robot

    def reset(self) -> None:
        self.commander.reset()

    def run(self):
        # listener = Listener(on_press=self.on_press)
        # listener.start()
        self.ps4_manager.start()

        while True:
            time.sleep(self.freq)
            # activated_keys, command_2d = self.process_command()
            # if "q" in activated_keys:
            #    break
            if not self.ps4_manager.is_running:
                break
            command_2d = self.ps4_manager.joy_vector
            if command_2d is not None and np.linalg.norm(command_2d) > 1e-3:
                command_3d = np.array([command_2d[0], command_2d[1], 0.0])
                try:
                    self.robot.move_end_pos(command_3d, wrt="world")
                    joint_angles_next = self.robot.get_joint_angles()
                    self.commander.send_command(joint_angles_next)

                    if self.post_command_hook is not None:
                        self.post_command_hook(self)

                except IKFailError:
                    print("IK fail!")
