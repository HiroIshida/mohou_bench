import argparse
import pickle
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple, Type

import numpy as np
import pybullet as pb
import pybullet_data
from bunsetsu.file import get_segmentation_path
from bunsetsu.propagator import SequentialPropagator
from mohou.default import create_default_propagator
from mohou.file import get_project_path
from mohou.propagator import Propagator
from mohou.types import (
    AngleVector,
    ElementDict,
    EpisodeBundle,
    EpisodeData,
    MetaData,
    RGBImage,
)
from skrobot.coordinates.math import rpy2quaternion, wxyz2xyzw

from mohou_bench.camera import Camera
from mohou_bench.commander import Commander
from mohou_bench.pybullet_utils import (
    BoxConfig,
    CylinderConfig,
    PrimitiveConfig,
    PybulletColor,
)
from mohou_bench.robot import (
    BoxStickPandaModel,
    CylinderStickPandaModel,
    LboxStickPandaModel,
    StickPandaModelBase,
)
from mohou_bench.teleop import PS4Button, TeleoperationCommander
from mohou_bench.utils import PropagatorLike, get_skrobot_coords


@dataclass
class CylinderPositionRandomizer:
    radius: float = 0.03
    n_cylinder: int = 3
    std = 0.06
    rand_center: np.ndarray = np.array((0.45, 0.0))
    keep_distance: bool = False

    def create_table(self) -> Dict[str, np.ndarray]:

        center_table: Dict[str, np.ndarray] = {}
        if self.keep_distance:
            threshold = 0.02
        else:
            threshold = 0.0

        def is_valid_position(query: np.ndarray) -> bool:
            if np.linalg.norm(query[:2] - self.rand_center) > 0.125:
                return False

            for center in center_table.values():
                if np.linalg.norm(query - center) < self.radius * 2 + 1e-3 + threshold:
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


@dataclass
class GoalRegion:
    width: float
    depth: float
    center: np.ndarray

    def create(self):
        x_center, y_center = self.center
        conf = BoxConfig(size=(self.depth, 0.02, 0.02), rgba="gray")
        conf.to_pybullet_object(pos=(x_center, y_center - 0.5 * self.width), fixed=True)

        conf = BoxConfig(size=(self.depth, 0.02, 0.02), rgba="gray")
        conf.to_pybullet_object(pos=(x_center, y_center + 0.5 * self.width), fixed=True)

        conf = BoxConfig(size=(0.02, self.width, 0.02), rgba="gray")
        conf.to_pybullet_object(pos=(x_center + self.depth * 0.5 - 0.01, y_center), fixed=True)

    def is_inside(self, x: np.ndarray) -> bool:
        abs_diff = np.abs(x - self.center)
        if abs_diff[0] > self.depth * 0.5:
            return False
        if abs_diff[1] > self.width * 0.5:
            return False
        return True


class WorldBase(ABC):
    id_table: Dict[str, int]
    center_table: Dict[str, np.ndarray]
    color_table: Dict[str, PybulletColor]
    randomizer: CylinderPositionRandomizer

    def __init__(
        self, n_cylinder: int, keep_distance: bool = False, use_single_color: bool = False
    ):
        conf: PrimitiveConfig

        radius = 0.03
        randomizer = CylinderPositionRandomizer(
            radius=radius, n_cylinder=n_cylinder, keep_distance=keep_distance
        )
        center_dict = randomizer.create_table()

        if use_single_color:
            color_list = [PybulletColor.red]
        else:
            color_list = [PybulletColor.red, PybulletColor.green, PybulletColor.blue]

        id_table = {}
        color_table = {}
        for idx, name in enumerate(center_dict.keys()):
            color = color_list[idx % len(color_list)]
            conf = CylinderConfig(radius=radius, height=0.02, rgba=color)
            center = center_dict[name]
            object_id = conf.to_pybullet_object(pos=list(center))
            id_table[name] = object_id
            color_table[name] = color

        self.id_table = id_table
        self.color_table = color_table
        self.center_table = center_dict
        self.randomizer = randomizer
        self._post_init_hook()

    @abstractmethod
    def _post_init_hook(self) -> None:
        pass

    @abstractmethod
    def is_successful(self) -> bool:
        pass

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


class SingleGoalWorld(WorldBase):
    goal_region: GoalRegion

    def _post_init_hook(self) -> None:
        goal_region = GoalRegion(0.18, 0.15, np.array([0.7, -0.2]))
        goal_region.create()
        self.goal_region = goal_region

    def is_successful(self) -> bool:
        for body_id in self.id_table.values():
            co = get_skrobot_coords(body_id)
            pos = co.translation[:2]
            if not self.goal_region.is_inside(pos):
                return False
        return True


class DualGoalWorld(WorldBase):
    goal_region_left: GoalRegion
    goal_region_right: GoalRegion

    def _post_init_hook(self) -> None:
        goal_region_left = GoalRegion(0.18, 0.15, np.array([0.7, -0.2]))
        goal_region_right = GoalRegion(0.18, 0.15, np.array([0.7, +0.2]))
        goal_region_left.create()
        goal_region_right.create()
        self.goal_region_left = goal_region_left
        self.goal_region_right = goal_region_right

    def is_successful(self) -> bool:
        for key in self.id_table.keys():
            color = self.color_table[key]
            body_id = self.id_table[key]
            co = get_skrobot_coords(body_id)
            pos = co.translation[:2]
            if color == PybulletColor.red:
                if not self.goal_region_left.is_inside(pos):
                    return False
            elif color == PybulletColor.green:
                if not self.goal_region_right.is_inside(pos):
                    return False
            else:
                assert False
        return True


def filter_episode(episode_list: List[EpisodeData]):
    av_init_list = []
    for episode in episode_list:
        av_seq = episode.get_sequence_by_type(AngleVector)
        av_init = av_seq[0].numpy()
        av_init_list.append(av_init)
    av_init_arr = np.array(av_init_list)
    av_median = np.median(av_init_arr, axis=0)
    dists = np.sqrt(np.sum((av_init_arr - av_median) ** 2, axis=1))
    valid_indices = np.where(dists < 0.1)[0]
    return [episode_list[idx] for idx in valid_indices]


class Mode(Enum):
    dataset = 0
    test = 1
    sampling = 2


class WorldMode(Enum):
    single = SingleGoalWorld
    dual = DualGoalWorld


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", type=int, default=2, help="number of object")
    parser.add_argument("-mode", type=str, default="dataset", help="mode")
    parser.add_argument("-stick", type=str, default="box", help="stick model")
    parser.add_argument("-world", type=str, default="single", help="single goal world")
    parser.add_argument("--single_color", action="store_true", help="use single color")
    parser.add_argument("--bunsetsu", action="store_true", help="use segmented policy")
    args = parser.parse_args()
    mode_str: str = args.mode
    n_cylinder: int = args.m
    stick_model: str = args.stick
    use_single_color: bool = args.single_color
    use_bunsetsu: bool = args.bunsetsu
    world_mode_str: str = args.world

    robot_type: Type[StickPandaModelBase]
    if stick_model == "box":
        robot_type = BoxStickPandaModel
    elif stick_model == "cylinder":
        robot_type = CylinderStickPandaModel
    elif stick_model == "lbox":
        robot_type = LboxStickPandaModel
    else:
        assert False

    mode = Mode[mode_str]
    project_name = "push_{}stick_{}cylinder".format(stick_model, n_cylinder)
    if use_single_color:
        project_name += "_singlecolor"
    if world_mode_str != "single":
        project_name += "_{}".format(world_mode_str)

    project_path = get_project_path(project_name)
    project_path.mkdir(exist_ok=True)

    episod_tmp_dir = project_path / "tmp_episode_cache"
    episod_tmp_dir.mkdir(exist_ok=True)

    pb.connect(pb.GUI)
    pb.configureDebugVisualizer(pb.COV_ENABLE_GUI, 0)
    pb.setGravity(0, 0, -10)
    pb.setAdditionalSearchPath(pybullet_data.getDataPath())  # used by loadURDF
    pb.loadURDF("plane.urdf")
    world_type = WorldMode[world_mode_str].value
    world = world_type(n_cylinder, keep_distance=True, use_single_color=use_single_color)

    camera = Camera.create(Camera.CameraPosition.frontclose)

    if mode == Mode.test:
        prop: PropagatorLike
        if use_bunsetsu:
            sp = get_segmentation_path(project_path, "arhmm2")
            prop = SequentialPropagator.load(sp)
        else:
            prop = create_default_propagator(project_path, Propagator)
        raw_com = Commander.create(robot_type=robot_type)

        success_count = 0
        episode_list = []
        for _ in range(100):
            prop.reset()
            world.reset(randomize=True)
            raw_com.reset()

            edict_list = []
            for _ in range(250):
                av = AngleVector(raw_com.ri.get_joint_angles())
                rgb = RGBImage(camera.render())
                edict = ElementDict([av, rgb])
                edict_list.append(edict)
                prop.feed(edict)

                edict_next = prop.predict(1)[0]
                av_next = edict_next[AngleVector]
                raw_com.send_command(av_next.numpy())

            # post trial
            is_successful = world.is_successful()
            episode_metadata = MetaData({"success": is_successful})
            episode = EpisodeData.from_edict_list(
                edict_list, metadata=episode_metadata, check_terminate_flag=False
            )
            episode_list.append(episode)
            if is_successful:
                success_count += 1
            n_trial = len(episode_list)
            success_rate = success_count / float(n_trial)
            print("trail: {}, success rate: {}".format(n_trial, success_rate))

        metadata = MetaData({"success_rate": success_count / float(len(episode_list))})
        bundle = EpisodeBundle.from_episodes(episode_list, meta_data=metadata)
        sampling_result_path = project_path / "sampling_result"
        sampling_result_path.mkdir(exist_ok=True)
        bundle.dump(sampling_result_path, postfix=str(uuid.uuid4()), compress=True)

    elif mode == Mode.dataset:
        edict_list: List[ElementDict] = []  # type: ignore

        com = TeleoperationCommander.create(robot_type=robot_type)

        def post_command_hook(com: TeleoperationCommander):
            av = AngleVector(com.robot.get_joint_angles())
            rgb = RGBImage(camera.render())
            edict_list.append(ElementDict([av, rgb]))

        def reset_callback(save_episode: bool = True):
            global edict_list
            print("success? {}".format(world.is_successful()))
            world.reset(randomize=True)
            com.reset()
            print("sequence length {}".format(len(edict_list)))

            if save_episode:
                if len(edict_list) > 10:
                    episode = EpisodeData.from_edict_list(edict_list)
                    episode_cache_path = episod_tmp_dir / "{}.pkl".format(uuid.uuid4())
                    with episode_cache_path.open(mode="wb") as f:
                        pickle.dump(episode, f)
                    print("saved episode to {}".format(episode_cache_path))
                    n_episode = len(list(episod_tmp_dir.iterdir()))
                    print("current episode number: {}".format(n_episode))
            else:
                print("episode is not saved")
            edict_list = []

        com.ps4_manager.register_callback(PS4Button.CIRCLE, lambda: reset_callback())
        com.ps4_manager.register_callback(
            PS4Button.CROSS, lambda: reset_callback(save_episode=False)
        )
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
            episode_list = filter_episode(episode_list)
            bundle = EpisodeBundle.from_episodes(episode_list)
            bundle.dump(project_path, compress=True, exist_ok=True)
            bundle.plot_vector_histories(AngleVector, project_path)
    else:
        assert False
