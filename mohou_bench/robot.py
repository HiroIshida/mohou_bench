import copy
import subprocess
import time
from abc import ABC, abstractmethod
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable, Dict, List, Optional

import gdown
import numpy as np
import pybullet as pb
from skrobot.coordinates import Coordinates
from skrobot.model import RobotModel
from skrobot.models.urdf import RobotModelFromURDF


def get_cache_path() -> Path:
    p = Path("~/.cache/mohou_bench").expanduser()
    p.mkdir(exist_ok=True)
    return p


class IKFailError(Exception):
    pass


class PandaModelBase(ABC):
    robot_model: RobotModel

    def __init__(self):
        urdf_path = self.get_urdf_path()
        model = RobotModelFromURDF(urdf_file=str(urdf_path))
        self.robot_model = model

    @classmethod
    @abstractmethod
    def get_urdf_path(cls) -> Path:
        pass

    @abstractmethod
    def get_end_effector_name(self) -> str:
        pass

    def solve_ik(self, coords: Coordinates) -> None:
        joints = [self.robot_model.__dict__[jname] for jname in self.control_joint_names()]
        link_list = [joint.child_link for joint in joints]

        end_effector = self.get_end_effector()
        av_next = self.robot_model.inverse_kinematics(coords, end_effector, link_list)
        solved = isinstance(av_next, np.ndarray)
        if not solved:
            raise IKFailError

    def set_joint_angles(self, angles: List[float]):
        for name, angle in zip(self.control_joint_names(), angles):
            joint = self.robot_model.__dict__[name]
            joint.joint_angle(angle)

    def get_joint_angles(self) -> np.ndarray:
        angles = []
        for joint_name in self.control_joint_names():
            joint = self.robot_model.__dict__[joint_name]
            angles.append(joint.joint_angle())
        return np.array(angles)

    def control_joint_names(self) -> List[str]:
        return ["panda_joint{}".format(i + 1) for i in range(7)]

    def get_end_effector(self) -> Coordinates:
        ef_name = self.get_end_effector_name()
        end_effector = self.robot_model.__dict__[ef_name]
        return end_effector

    def move_end_pos(self, pos, wrt: str = "local") -> None:
        co_end_effector = self.get_end_effector().copy_worldcoords()
        pos = np.array(pos, dtype=np.float64)
        co_end_effector.translate(pos, wrt=wrt)
        self.solve_ik(co_end_effector)

    def move_end_rot(self, angle, axis, wrt: str = "local") -> None:
        co_end_effector = self.get_end_effector().copy_worldcoords()
        co_end_effector.rotate(angle, axis, wrt=wrt)
        self.solve_ik(co_end_effector)


class StickPandaModel(PandaModelBase):
    @classmethod
    def get_urdf_path(cls) -> Path:
        url = "https://drive.google.com/uc?id=1uuCFJjCqkHQcGts3lSOwhqzACfYmlWNM"

        urdf_path = get_cache_path() / "franka_panda_stick" / "panda.urdf"
        if urdf_path.exists():
            return urdf_path

        with TemporaryDirectory() as f:
            tar_path = Path(f) / "tmp.tar"
            gdown.download(url, str(tar_path), quiet=False)

            cmd = "cd {cache_path} && tar xf {tar_path}".format(
                cache_path=get_cache_path(), tar_path=tar_path
            )
            subprocess.run(cmd, shell=True)
        return urdf_path

    def get_end_effector_name(self) -> str:
        return "panda_grasptarget"

    def init_pose(self):
        joint_angles = [0.7, 0.7, 0.0, -0.5, 0.0, 1.3, -0.8]
        self.set_joint_angles(joint_angles)


class PybulletRobotInterface:
    model: PandaModelBase
    robot_id: int
    joint_table: Dict[str, int]
    link_table: Dict[str, int]
    latest_commands: Dict[str, float]
    default_callback: Optional[Callable] = None

    def __init__(self, robot_model: PandaModelBase):
        path = robot_model.get_urdf_path()
        robot_id = pb.loadURDF(str(path), useFixedBase=True)

        joint_table = {}
        link_table = {pb.getBodyInfo(robot_id)[0].decode("UTF-8"): -1}
        for idx in range(pb.getNumJoints(robot_id)):
            joint_info = pb.getJointInfo(robot_id, idx)
            joint_id = joint_info[0]
            joint_name = joint_info[1].decode("UTF-8")
            joint_table[joint_name] = joint_id

            tmp = joint_info[12].decode("UTF-8")
            name = "_".join(tmp.split("/"))
            link_table[name] = idx

        self.model = copy.deepcopy(robot_model)
        self.robot_id = robot_id
        self.joint_table = joint_table
        self.link_table = link_table
        self.latest_commands = {}

    def reset(self):
        self.latest_commands = {}

    def command_angle(self, joint_name: str, angle: float, gain: float = 1.0, force: float = 300):
        self.latest_commands[joint_name] = angle

        joint_id = self.joint_table[joint_name]
        pb.setJointMotorControl2(
            bodyIndex=self.robot_id,
            jointIndex=joint_id,
            controlMode=pb.POSITION_CONTROL,
            targetPosition=angle,
            targetVelocity=0.0,
            force=force,
            positionGain=gain,
            velocityGain=1.0,
            maxVelocity=1.0,
        )

    def command_angles(self, robot: PandaModelBase, gain: float = 1.0):
        for name in robot.control_joint_names():
            joint = robot.robot_model.__dict__[name]
            self.command_angle(name, joint.joint_angle(), gain=gain)

    def reset_angles(self, robot: PandaModelBase):
        for name in robot.control_joint_names():
            joint = robot.robot_model.__dict__[name]
            joint_id = self.joint_table[name]
            pb.resetJointState(self.robot_id, joint_id, joint.joint_angle())

    def wait_interpolation(
        self, sleep: float = 0.0, callback: Optional[Callable] = None, vel_threshold: float = 0.05
    ) -> None:
        while True:
            self.step(1, sleep, callback=callback)
            velocities = []
            for joint_id in self.joint_table.values():
                _, vel, _, _ = pb.getJointState(self.robot_id, joint_id)
                velocities.append(vel)
            vel_max = np.max(np.abs(velocities))
            if vel_max < vel_threshold:
                break

    def step(self, n: int, sleep: float = 0.0, callback: Optional[Callable] = None) -> None:
        for _ in range(n):
            pb.stepSimulation()

            if callback is not None:
                callback(self)
            else:
                if self.default_callback is not None:
                    self.default_callback(self)
            time.sleep(sleep)

    def get_joint_angles(self) -> np.ndarray:
        joint_names = self.model.control_joint_names()

        angle_list = []
        for name in joint_names:
            joint_id = self.joint_table[name]
            pos, _, _, _ = pb.getJointState(self.robot_id, joint_id)
            angle_list.append(pos)
        return np.array(angle_list)
