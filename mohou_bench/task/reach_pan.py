import time
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pybullet as pb
import pybullet_data
from skrobot.coordinates.math import rpy2quaternion, wxyz2xyzw

from mohou_bench.asset import get_fryingpan_urdf_path
from mohou_bench.commander import Commander


@dataclass
class World:
    id_table: Dict[str, int]
    configuration_table: Dict[str, np.ndarray]

    @classmethod
    def create(cls):
        frypan_path_str = get_fryingpan_urdf_path()
        pan_id = pb.loadURDF(str(frypan_path_str))
        id_table = {"pan": pan_id}

        center_table = {"pan": np.array([0.5, 0.0, np.pi * 0.5])}
        return cls(id_table, center_table)

    def set_pose(
        self,
        body_name: str,
        point: Tuple[float, float, float],
        rpy: Tuple[float, float, float] = (0, 0, 0),
    ):
        body_id = self.id_table[body_name]
        q = rpy2quaternion(rpy)
        pb.resetBasePositionAndOrientation(body_id, point, wxyz2xyzw(q))
        pb.resetBaseVelocity(
            body_id,
            linearVelocity=(0.0, 0.0, 0.0),
            angularVelocity=(0.0, 0.0, 0.0),
        )

    def reset(self, randomize: bool = False):
        if randomize:
            self.randomize()

        for key in self.id_table.keys():
            c = self.configuration_table[key]
            point = (c[0], c[1], 0.0)
            rpy = (c[2], 0.0, 0.0)
            self.set_pose(key, point, rpy)  # type: ignore

    def randomize(self) -> None:
        c_nominal = np.array([0.6, 0.0, np.pi * 0.5])
        width = np.array([0.2, 0.2, 0.6])
        c = c_nominal - width * 0.5 + np.random.rand(3) * width
        print(c)
        self.configuration_table["pan"] = c


CLIENT = pb.connect(pb.GUI)
pb.setAdditionalSearchPath(pybullet_data.getDataPath())  # used by loadURDF
pb.configureDebugVisualizer(pb.COV_ENABLE_GUI, 0)
pb.configureDebugVisualizer(pb.COV_ENABLE_SHADOWS, 0)
pb.setGravity(0, 0, -10)
w = World.create()
w.reset()
com = Commander.create()

for i in range(10):
    w.reset(randomize=True)
    time.sleep(3)
