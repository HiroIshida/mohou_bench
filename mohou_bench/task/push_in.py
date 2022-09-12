import pickle
import uuid
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pybullet as pb
import pybullet_data
from mohou.file import get_project_path
from mohou.types import AngleVector, ElementDict, EpisodeBundle, EpisodeData, RGBImage
from skrobot.coordinates.math import rpy2quaternion, wxyz2xyzw

from mohou_bench.camera import Camera
from mohou_bench.pybullet_utils import (
    BoxConfig,
    CylinderConfig,
    PrimitiveConfig,
    PybulletColor,
)
from mohou_bench.teleop import PS4Button, TeleoperationCommander


@dataclass
class CylinderPositionRandomizer:
    radius: float = 0.03
    n_cylinder: int = 3
    std = 0.06
    rand_center: np.ndarray = np.array((0.5, 0.0))

    def create_table(self) -> Dict[str, np.ndarray]:

        center_table: Dict[str, np.ndarray] = {}

        def is_valid_position(query: np.ndarray) -> bool:
            if np.linalg.norm(query[:2] - self.rand_center) > 0.125:
                return False

            for center in center_table.values():
                if np.linalg.norm(query - center) < self.radius * 2 + 1e-3:
                    return False
            return True

        idx = 0
        while idx < self.n_cylinder:
            center_cand = np.hstack(
                [self.rand_center + np.random.randn(2) * self.std, [0.01 + 1e-3]]
            )
            if is_valid_position(center_cand):
                center_table["cylinder{}".format(idx)] = center_cand
                idx += 1
        return center_table


def get_yes_no():
    key = input()
    if key in ["y", "n"]:
        return key
    return get_yes_no()


class World:
    id_table: Dict[str, int]
    center_table: Dict[str, np.ndarray]
    randomizer: CylinderPositionRandomizer

    def __init__(self, n_cylinder: int):
        conf: PrimitiveConfig

        radius = 0.03
        randomizer = CylinderPositionRandomizer(radius=radius, n_cylinder=n_cylinder)
        center_dict = randomizer.create_table()

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
        self.randomizer = randomizer

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

    def reset(self, randomize: bool = False):
        if randomize:
            self.center_table = self.randomizer.create_table()
        for key in self.id_table.keys():
            center = self.center_table[key]
            self.set_pose(key, tuple(list(center)))  # type: ignore


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", type=int, default=2, help="number of object")
    args = parser.parse_args()
    n_cylinder: int = args.m

    project_name = "push_cylinder{}".format(n_cylinder)
    project_path = get_project_path(project_name)
    project_path.mkdir(exist_ok=True)

    episod_tmp_dir = project_path / "tmp_episode_cache"
    episod_tmp_dir.mkdir(exist_ok=True)

    pb.connect(pb.GUI)
    pb.configureDebugVisualizer(pb.COV_ENABLE_GUI, 0)
    pb.setGravity(0, 0, -10)
    pb.setAdditionalSearchPath(pybullet_data.getDataPath())  # used by loadURDF
    pb.loadURDF("plane.urdf")
    world = World(n_cylinder)

    camera = Camera.create(Camera.CameraPosition.frontclose)
    edict_list: List[ElementDict] = []

    com = TeleoperationCommander.create()

    def post_command_hook(com: TeleoperationCommander):
        av = AngleVector(com.robot.get_joint_angles())
        rgb = RGBImage(camera.render())
        edict_list.append(ElementDict([av, rgb]))

    def reset_callback():
        global edict_list
        world.reset(randomize=True)
        com.reset()
        print("sequence length {}".format(len(edict_list)))

        if len(edict_list) > 10:
            episode = EpisodeData.from_edict_list(edict_list)
            episode_cache_path = episod_tmp_dir / "{}.pkl".format(uuid.uuid4())
            with episode_cache_path.open(mode="wb") as f:
                pickle.dump(episode, f)
            print("saved episode to {}".format(episode_cache_path))
            edict_list = []

    com.ps4_manager.register_callback(PS4Button.R1, lambda: reset_callback())
    com.post_command_hook = post_command_hook
    com.run()

    print("create episode bundle? [y/n]")
    key = get_yes_no()
    if key == "y":
        episode_list = []
        for file_path in episod_tmp_dir.iterdir():
            with file_path.open("rb") as f:
                episode = pickle.load(f)
            episode_list.append(episode)
        bundle = EpisodeBundle.from_episodes(episode_list)
        bundle.dump(project_path, compress=True, exist_ok=True)
        bundle.plot_vector_histories(AngleVector, project_path)
