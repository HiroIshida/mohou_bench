import copy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pybullet as pb
import pybullet_data
import tqdm
from mohou.file import get_project_path
from mohou.types import AngleVector, ElementDict, EpisodeBundle, EpisodeData, MetaData
from skrobot.coordinates import Coordinates
from skrobot.coordinates.math import (
    quaternion2matrix,
    rpy2quaternion,
    wxyz2xyzw,
    xyzw2wxyz,
)

from mohou_bench.asset import get_fryingpan_urdf_path
from mohou_bench.camera import Camera
from mohou_bench.commander import Commander
from mohou_bench.pybullet_utils import create_debug_axis
from mohou_bench.robot import GripperPandaModel


@dataclass
class World:
    id_table: Dict[str, int]
    configuration_table: Dict[str, np.ndarray]

    @classmethod
    def create(cls):
        plane_id = pb.loadURDF("plane.urdf")

        frypan_path_str = get_fryingpan_urdf_path()
        pan_id = pb.loadURDF(str(frypan_path_str))
        id_table = {"pan": pan_id, "plane": plane_id}

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

    def get_skrobot_coords(self, body_name: str) -> Coordinates:
        # NOTE quat is xyzw order
        body_id = self.id_table[body_name]
        trans, quat = pb.getBasePositionAndOrientation(body_id)
        mat = quaternion2matrix(xyzw2wxyz(quat))
        return Coordinates(trans, mat)

    def reset(self, randomize: bool = False, configuration: Optional[np.ndarray] = None):

        if configuration is not None:
            assert not randomize
            self.configuration_table["pan"] = configuration

        if randomize:
            self.randomize()

        key = "pan"
        c = self.configuration_table[key]
        point = (c[0], c[1], 0.03)
        rpy = (c[2], 0.0, 0.0)
        self.set_pose(key, point, rpy)  # type: ignore

    def randomize(self) -> None:
        c_nominal = np.array([0.5, 0.0, np.pi * 0.5])
        width = np.array([0.2, 0.2, 0.6])
        c = c_nominal - width * 0.5 + np.random.rand(3) * width
        self.configuration_table["pan"] = c


def oracle_rollout(commander: Commander, world: World, camera: Camera) -> EpisodeData:
    robot_model = copy.deepcopy(commander.robot)

    av_init = robot_model.get_joint_angles()

    co = world.get_skrobot_coords("pan")
    co.translate([0.15, 0.0, 0.15])
    co.rotate(np.pi * 0.5, "y")
    co.rotate(np.pi * 1.0, "x")
    create_debug_axis(co)

    robot_model.solve_ik(co)
    av_pre_grasp = robot_model.get_joint_angles()

    co.translate([0.0, 0.0, -0.12], "world")
    robot_model.solve_ik(co)
    av_grasp = robot_model.get_joint_angles()

    n_point_pregrasp = 70
    n_point_grasp = 30
    width_pregrasp = (av_pre_grasp - av_init) / (n_point_pregrasp - 1)  # type: ignore
    av_list = [av_init + width_pregrasp * i for i in range(n_point_pregrasp)]
    av_list.pop()

    width_grasp = (av_grasp - av_pre_grasp) / (n_point_grasp - 1)  # type: ignore
    av_list.extend([av_pre_grasp + width_grasp * i for i in range(n_point_grasp)])

    edict_list = []
    n_command_split = 3
    for av in av_list:
        robot_model.set_joint_angles(av)
        render_result = camera.render()
        commander.send_command(robot_model, n_command_split=n_command_split)
        mohou_av = AngleVector(av)
        edict = ElementDict([mohou_av, render_result.mohou_rgb, render_result.mohou_segmentation])
        edict_list.append(edict)

    metadata = MetaData(
        {"n_command_split": n_command_split, "step_length": commander.default_step_length}
    )
    episode = EpisodeData.from_edict_list(edict_list, metadata=metadata)
    return episode


def reset(
    commander: Commander,
    world: World,
    randomize: bool = False,
    configuration: Optional[np.ndarray] = None,
):
    commander.reset()
    robot_model: GripperPandaModel = copy.deepcopy(commander.robot)  # type: ignore
    robot_model.move_end_pos([0.05, 0.0, 0.2], wrt="world")
    robot_model.set_gripper_joints(np.array([0.04, 0.04]))
    commander.send_command(robot_model)
    world.reset(randomize=randomize, configuration=configuration)


def get_regular_grid_coords() -> List[np.ndarray]:
    def gen(n_dim: int, n_split: int) -> np.ndarray:
        assert n_dim > 0
        arr = np.expand_dims(np.linspace(0, 1, n_split), axis=0).T
        for i in range(n_dim - 1):
            row, col = arr.shape
            partial_list = []
            for val in np.linspace(0, 1, n_split):
                partial = np.ones((row, col + 1)) * val
                partial[:, 1:] = arr
                partial_list.append(partial)
            arr = np.vstack(partial_list)
        return arr

    center = [0.5, 0.0, np.pi * 0.5]
    width = np.array([0.2, 0.2, 0.6])
    b_min = center - 0.5 * width

    coords_list = []
    for index_like in gen(3, 3):
        coords_list.append(b_min + width * index_like)
    return coords_list


if __name__ == "__main__":
    project_name = "reach_pan"
    project_path = get_project_path(project_name)
    project_path.mkdir(exist_ok=True)

    CLIENT = pb.connect(pb.DIRECT)
    pb.setAdditionalSearchPath(pybullet_data.getDataPath())  # used by loadURDF
    pb.configureDebugVisualizer(pb.COV_ENABLE_GUI, 0)
    pb.configureDebugVisualizer(pb.COV_ENABLE_SHADOWS, 0)
    pb.setGravity(0, 0, -10)
    com = Commander.create(robot_type=GripperPandaModel)
    world = World.create()

    camera = Camera.create(Camera.CameraPosition.rightfront, n_pixel=112)
    episode_list = []
    for coords in tqdm.tqdm(get_regular_grid_coords()):
        reset(com, world, randomize=False, configuration=coords)
        episode = oracle_rollout(com, world, camera)
        episode_list.append(episode)

    untouch_episode_list = []
    for _ in tqdm.tqdm(range(10)):
        reset(com, world, randomize=True)
        episode = oracle_rollout(com, world, camera)
        untouch_episode_list.append(episode)

    bundle = EpisodeBundle(episode_list, untouch_episode_list, MetaData({}))
    bundle.dump(project_path, exist_ok=True)
    bundle.plot_vector_histories(AngleVector, project_path)
