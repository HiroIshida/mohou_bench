import numpy as np
import pybullet as pb
from skrobot.coordinates import Coordinates

from mohou_bench.robot import CylinderStickPandaModel, PybulletRobotInterface
from mohou_bench.utils import is_close


def test_stick_panda_model() -> None:
    model = CylinderStickPandaModel()
    model.init_pose()
    target = Coordinates(pos=(0.4, 0.0, 0.3))

    target.rotate(np.pi * 0.5, "y")
    target.rotate(np.pi * 0.5, "x")
    model.solve_ik(target)

    assert is_close(model.get_end_effector(), target)


def test_pybullet_robot_interface() -> None:
    pb.connect(pb.DIRECT)

    model = CylinderStickPandaModel()
    ri = PybulletRobotInterface(model)

    model.init_pose()
    ri.command_angles(model)
    ri.wait_interpolation()
    ri_angles = ri.get_joint_angles()
    model_angles = model.get_joint_angles()
    np.testing.assert_almost_equal(model_angles, ri_angles, decimal=3)

    model.init_pose()
    ri.reset_angles(model)
    ri_angles = ri.get_joint_angles()
    np.testing.assert_almost_equal(model.get_joint_angles(), ri_angles, decimal=3)
