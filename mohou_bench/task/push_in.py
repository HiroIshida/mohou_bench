from typing import Dict, List, Tuple

import numpy as np
import pybullet as pb
import pybullet_data
from mohou.types import AngleVector, ElementDict, EpisodeData, RGBImage
from skrobot.coordinates.math import rpy2quaternion, wxyz2xyzw

from mohou_bench.camera import Camera
from mohou_bench.pybullet_utils import (
    BoxConfig,
    CylinderConfig,
    PrimitiveConfig,
    PybulletColor,
)
from mohou_bench.teleop import PS4Button, TeleoperationCommander


def determine_cylinder_pos(
    n: int, r: float, rand_center: np.ndarray, std=0.03
) -> Dict[str, np.ndarray]:
    assert len(rand_center) == 2

    center_table: Dict[str, np.ndarray] = {}

    def is_valid_position(query: np.ndarray) -> bool:
        for center in center_table.values():
            if np.linalg.norm(query - center) < r * 2 + 1e-3:
                return False
        return True

    idx = 0
    while idx < n:
        center_cand = np.hstack([rand_center + np.random.randn(2) * std, [0.01 + 1e-3]])
        if is_valid_position(center_cand):
            center_table["cylinder{}".format(idx)] = center_cand
            idx += 1
    return center_table


class World:
    id_table: Dict[str, int]
    center_table: Dict[str, np.ndarray]

    def __init__(self):
        n_cylinder = 3
        conf: PrimitiveConfig
        radius = 0.03
        center_dict = determine_cylinder_pos(n_cylinder, radius, np.array((0.5, 0.0)))

        color_list = [PybulletColor.red, PybulletColor.green, PybulletColor.blue]
        id_table = {}
        for idx, name in enumerate(center_dict.keys()):
            color = color_list[idx % len(color_list)]
            conf = CylinderConfig(radius=radius, height=0.02, rgba=color)
            center = center_dict[name]
            object_id = conf.to_pybullet_object(pos=list(center))
            id_table[name] = object_id

        self.id_table = id_table
        self.center_table = center_dict

        # create goal region
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

    def reset(self):
        for key in self.id_table.keys():
            center = self.center_table[key]
            self.set_pose(key, tuple(list(center)))  # type: ignore


if __name__ == "__main__":
    pb.connect(pb.GUI)
    pb.configureDebugVisualizer(pb.COV_ENABLE_GUI, 0)
    pb.setGravity(0, 0, -10)
    pb.setAdditionalSearchPath(pybullet_data.getDataPath())  # used by loadURDF
    pb.loadURDF("plane.urdf")
    world = World()

    camera = Camera.create(Camera.CameraPosition.front)
    edict_list: List[ElementDict] = []

    com = TeleoperationCommander.create()

    def post_command_hook(com: TeleoperationCommander):
        av = AngleVector(com.robot.get_joint_angles())
        rgb = RGBImage(camera.render())
        edict_list.append(ElementDict([av, rgb]))

    def reset_callback():
        world.reset()
        com.reset()

    com.ps4_manager.register_callback(PS4Button.R1, lambda: reset_callback())
    com.post_command_hook = post_command_hook
    com.run()
    episode = EpisodeData.from_edict_list(edict_list)
