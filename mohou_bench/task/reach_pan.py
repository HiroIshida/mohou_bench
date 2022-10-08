import argparse
import copy
from enum import Enum
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pybullet as pb
import pybullet_data
import tqdm
from mohou.file import get_project_path
from mohou.propagator import LSTMPropagator
from mohou.types import AngleVector, ElementDict, EpisodeBundle, EpisodeData, MetaData

from mohou_bench.asset import BulletObject, FryingPanObject, PlaneObject
from mohou_bench.camera import Camera
from mohou_bench.commander import Commander
from mohou_bench.pybullet_utils import create_debug_axis
from mohou_bench.robot import GripperPandaModel
from mohou_bench.task_base import Task


class World(Task):
    @classmethod
    def create(cls):
        object_list: List[BulletObject] = []

        def pan_randomizer(pose_init: np.ndarray) -> np.ndarray:
            width = np.array([0.2, 0.2, 0.6])
            diff = -width * 0.5 + np.random.rand(3) * width
            pose_new = copy.deepcopy(pose_init)
            pose_new[0] += diff[0]
            pose_new[1] += diff[1]
            pose_new[3] += diff[2]
            return pose_new

        object_list.append(
            FryingPanObject.load(
                "pan", np.array([0.5, 0.0, 0.03, np.pi * 0.5, 0.0, 0.0]), randomizer=pan_randomizer
            )
        )
        object_list.append(PlaneObject.load())
        return cls(object_list)

    def reset(self, randomize: bool = False):
        if randomize:
            for obj in self.object_table.values():
                obj.randomize_pose()
        self.update_world()

    def set_configuration(self, vec: np.ndarray) -> None:
        assert len(vec) == 3
        diff = np.array([vec[0], vec[1], 0.0, vec[2], 0.0, 0.0])
        pose_new = self.object_table["pan"].init_pose + diff
        self.object_table["pan"].pose = pose_new
        self.update_world()


def oracle_rollout(commander: Commander, world: World, camera: Camera) -> EpisodeData:
    robot_model = copy.deepcopy(commander.robot)

    av_init = robot_model.get_joint_angles()

    co = world.get_skrobot_coords("pan")
    co.translate([0.15, 0.01, 0.15])
    co.rotate(np.pi * 0.5, "y")
    co.rotate(np.pi * 1.0, "x")
    create_debug_axis(co)

    robot_model.solve_ik(co)
    av_pre_grasp = robot_model.get_joint_angles()

    co.translate([0.0, 0.0, -0.16], "world")
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
        # edict = ElementDict([mohou_av, render_result.mohou_rgb, render_result.mohou_segmentation])
        edict = ElementDict([mohou_av, render_result.mohou_rgb])
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
    robot_model.set_gripper_joints(np.array([0.02, 0.02]))
    commander.send_command(robot_model)
    if configuration is None:
        world.reset(randomize=randomize)
    else:
        world.set_configuration(configuration)


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

    center = np.zeros(3)
    width = np.array([0.2, 0.2, 0.6])
    b_min = center - 0.5 * width

    coords_list = []
    for index_like in gen(3, 3):
        coords_list.append(b_min + width * index_like)
    return coords_list


class Mode(Enum):
    dataset = 1
    oneshot = 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-mode", type=str, default="dataset", help="mode")
    parser.add_argument("--gui", action="store_true", help="gui")
    args = parser.parse_args()
    mode_str: str = args.mode
    use_gui: bool = args.gui
    mode = Mode[mode_str]

    project_name = "reach_pan"
    project_path = get_project_path(project_name)
    project_path.mkdir(exist_ok=True)

    CLIENT = pb.connect(pb.GUI if use_gui else pb.DIRECT)
    pb.setAdditionalSearchPath(pybullet_data.getDataPath())  # used by loadURDF
    pb.configureDebugVisualizer(pb.COV_ENABLE_GUI, 0)
    pb.configureDebugVisualizer(pb.COV_ENABLE_SHADOWS, 0)
    pb.setGravity(0, 0, -10)
    com = Commander.create(robot_type=GripperPandaModel)
    world = World.create()
    camera = Camera.create(Camera.CameraPosition.rightfront, n_pixel=112)

    if mode == Mode.dataset:
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
    elif mode == Mode.oneshot:
        propagator = LSTMPropagator.create_default(project_path)
        assert not propagator.require_static_context

        np.random.seed(12345678)
        for i in tqdm.tqdm(range(10)):
            propagator.reset()
            reset(com, world, randomize=True)

            render_result = camera.render()
            rgb = render_result.mohou_rgb
            av = AngleVector(com.robot.get_joint_angles())
            ed = ElementDict([rgb, av])
            propagator.feed(ed)
            pred = propagator.predict(100, 0.9)

            av_pred = pred[-1][AngleVector]
            com.robot.set_joint_angles(list(av_pred.numpy()))
            com.send_command(com.robot)

            render_result = camera.render()
            fig, ax = plt.subplots()
            ax.imshow(render_result.mohou_rgb.numpy())
            plt.show()
