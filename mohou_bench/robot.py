import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List

import gdown
import numpy as np
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
        joints = [self.robot_model.__dict__[jname] for jname in self.control_joint_names]
        link_list = [joint.child_link for joint in joints]

        end_effector = self.get_end_effector()
        av_next = self.robot_model.inverse_kinematics(coords, end_effector, link_list)
        solved = isinstance(av_next, np.ndarray)
        if not solved:
            raise IKFailError

    def set_joint_angles(self, angles: List[float]):
        for name, angle in zip(self.control_joint_names, angles):
            joint = self.robot_model.__dict__[name]
            joint.joint_angle(angle)

    @property
    def control_joint_names(self) -> List[str]:
        return ["panda_joint{}".format(i + 1) for i in range(7)]

    def get_end_effector(self) -> Coordinates:
        ef_name = self.get_end_effector_name()
        end_effector = self.robot_model.__dict__[ef_name]
        return end_effector

    def move_end_pos(self, pos, wrt: str = "local") -> None:
        end_effector = self.get_end_effector()
        pos = np.array(pos, dtype=np.float64)
        end_effector.translate(pos, wrt=wrt)
        self.solve_ik(end_effector)

    def move_end_rot(self, angle, axis, wrt: str = "local") -> None:
        co_end_link = self.get_end_effector()
        co_end_link.rotate(angle, axis, wrt=wrt)
        self.solve_ik(co_end_link)


class StickPandaModel(PandaModelBase):
    @classmethod
    def get_urdf_path(cls) -> Path:
        url = "https://drive.google.com/uc?id=1rVKUPgZUpgsC8rboOExNXI0zybxytU1O"

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
