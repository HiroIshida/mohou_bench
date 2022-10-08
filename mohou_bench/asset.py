import subprocess
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable, Optional

import gdown
import numpy as np
import pybullet as pb
import pybullet_data


def get_cache_path() -> Path:
    p = Path("~/.cache/mohou_bench").expanduser()
    p.mkdir(exist_ok=True)
    return p


def get_fryingpan_urdf_path() -> Path:

    url = "https://drive.google.com/uc?id=1wtKEeOqKMg8h36LLqP64kzsgKUbtnDkA"

    urdf_path = get_cache_path() / "Chefmate_8_Frypan" / "object.urdf"

    if not urdf_path.exists():
        with TemporaryDirectory() as f:
            tar_path = Path(f) / "tmp.tar"
            gdown.download(url, str(tar_path), quiet=False)

            cmd = "cd {cache_path} && tar xf {tar_path}".format(
                cache_path=get_cache_path(), tar_path=tar_path
            )
            subprocess.run(cmd, shell=True)

    assert urdf_path.exists()
    return urdf_path


@dataclass
class BulletObject:
    name: str
    bullet_id: int
    init_pose: np.ndarray
    pose: np.ndarray
    randomizer: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]]

    def reset_pose(self) -> None:
        pass

    def randomize_pose(self) -> None:
        if self.randomizer is not None:
            self.pose = self.randomizer(self.init_pose)


@dataclass
class PlaneObject(BulletObject):
    @classmethod
    def load(cls) -> "PlaneObject":
        pb.setAdditionalSearchPath(pybullet_data.getDataPath())
        plane_id = pb.loadURDF("plane.urdf")
        return cls("plane", plane_id, np.zeros(6), np.zeros(6), None)


@dataclass
class FryingPanObject(BulletObject):
    @classmethod
    def load(
        cls, name: str, pose: np.ndarray, randomizer: Optional[Callable] = None
    ) -> "FryingPanObject":
        frypan_path_str = get_fryingpan_urdf_path()
        pan_id = pb.loadURDF(str(frypan_path_str))
        return cls(name, pan_id, pose, pose, randomizer)
