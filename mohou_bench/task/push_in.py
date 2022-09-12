from typing import List

import numpy as np
import pybullet as pb
import pybullet_data

from mohou_bench.pybullet_utils import (
    BoxConfig,
    CylinderConfig,
    PrimitiveConfig,
    PybulletColor,
)
from mohou_bench.teleop import TeleoperationCommander


def determine_cylinder_pos(n: int, r: float, rand_center: np.ndarray, std=0.03) -> List[np.ndarray]:
    assert len(rand_center) == 2

    center_list: List[np.ndarray] = []

    def is_valid_position(query: np.ndarray) -> bool:
        for center in center_list:
            if np.linalg.norm(query - center) < r * 2 + 1e-3:
                return False
        return True

    while len(center_list) < n:
        center_cand = rand_center + np.random.randn(2) * std
        if is_valid_position(center_cand):
            center_list.append(center_cand)
    return center_list


if __name__ == "__main__":
    pb.connect(pb.GUI)
    pb.configureDebugVisualizer(pb.COV_ENABLE_GUI, 0)
    pb.setGravity(0, 0, -10)
    pb.setAdditionalSearchPath(pybullet_data.getDataPath())  # used by loadURDF
    pb.loadURDF("plane.urdf")
    com = TeleoperationCommander.create()

    n_cylinder = 3

    conf: PrimitiveConfig

    radius = 0.03
    center_list = determine_cylinder_pos(n_cylinder, radius, np.array((0.5, 0.0)))

    color_list = [PybulletColor.red, PybulletColor.green, PybulletColor.blue]
    for idx, center in enumerate(center_list):
        color = color_list[idx % len(color_list)]
        conf = CylinderConfig(radius=radius, height=0.02, rgba=color)
        conf.to_pybullet_object(pos=list(center))

    width = 0.18
    depth = 0.15
    y_center = -0.2
    x_center = 0.7

    conf = BoxConfig(size=(depth, 0.02, 0.02), rgba="gray")
    conf.to_pybullet_object(pos=(x_center, y_center - 0.5 * width), fixed=True)

    conf = BoxConfig(size=(depth, 0.02, 0.02), rgba="gray")
    conf.to_pybullet_object(pos=(x_center, y_center + 0.5 * width), fixed=True)

    conf = BoxConfig(size=(0.02, width, 0.02), rgba="gray")
    conf.to_pybullet_object(pos=(x_center + depth * 0.5 - 0.01, y_center), fixed=True)

    com.run()
