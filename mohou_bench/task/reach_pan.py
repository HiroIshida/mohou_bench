import copy
import time
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pybullet as pb
import pybullet_data
from skrobot.coordinates import Coordinates
from skrobot.coordinates.math import (
    quaternion2matrix,
    rpy2quaternion,
    wxyz2xyzw,
    xyzw2wxyz,
)

from mohou_bench.asset import get_fryingpan_urdf_path
from mohou_bench.commander import Commander
from mohou_bench.pybullet_utils import create_debug_axis
from mohou_bench.robot import GripperPandaModel


@dataclass
class World:
    id_table: Dict[str, int]
    configuration_table: Dict[str, np.ndarray]

    @classmethod
    def create(cls):
        pb.loadURDF("plane.urdf")

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

    def get_skrobot_coords(self, body_name: str) -> Coordinates:
        # NOTE quat is xyzw order
        body_id = self.id_table[body_name]
        trans, quat = pb.getBasePositionAndOrientation(body_id)
        mat = quaternion2matrix(xyzw2wxyz(quat))
        return Coordinates(trans, mat)

    def reset(self, randomize: bool = False):
        if randomize:
            self.randomize()

        for key in self.id_table.keys():
            c = self.configuration_table[key]
            point = (c[0], c[1], 0.0)
            rpy = (c[2], 0.0, 0.0)
            self.set_pose(key, point, rpy)  # type: ignore

    def randomize(self) -> None:
        c_nominal = np.array([0.5, 0.0, np.pi * 0.5])
        width = np.array([0.2, 0.2, 0.6])
        c = c_nominal - width * 0.5 + np.random.rand(3) * width
        print(c)
        self.configuration_table["pan"] = c


def oracle_rollout(commander: Commander, world: World):
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

    for av in av_list:
        robot_model.set_joint_angles(av)
        commander.send_command(robot_model)
        time.sleep(0.01)


def reset(commander: Commander, world: World):
    commander.reset()
    robot_model: GripperPandaModel = copy.deepcopy(commander.robot)
    robot_model.move_end_pos([0.05, 0.0, 0.2], wrt="world")
    robot_model.set_gripper_joints([0.04, 0.04])
    commander.send_command(robot_model)
    world.reset()


CLIENT = pb.connect(pb.GUI)
pb.setAdditionalSearchPath(pybullet_data.getDataPath())  # used by loadURDF
pb.configureDebugVisualizer(pb.COV_ENABLE_GUI, 0)
pb.configureDebugVisualizer(pb.COV_ENABLE_SHADOWS, 0)
pb.setGravity(0, 0, -10)
com = Commander.create(robot_type=GripperPandaModel)
world = World.create()

reset(com, world)
oracle_rollout(com, world)
