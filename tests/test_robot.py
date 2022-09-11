import numpy as np
from skrobot.coordinates import Coordinates

from mohou_bench.robot import StickPandaModel
from mohou_bench.utils import is_close


def test_stick_panda_model():
    model = StickPandaModel()
    model.init_pose()
    target = Coordinates(pos=(0.4, 0.0, 0.3))
    target.rotate(np.pi * 0.5, "y")
    target.rotate(np.pi * 0.5, "x")
    model.solve_ik(target)

    assert is_close(model.get_end_effector(), target)
