import numpy as np
import pybullet as pb
from typing import Any, Optional, TypeVar, Generic, List, Tuple
from skrobot.coordinates import Coordinates
from skrobot.coordinates.math import quaternion2matrix, xyzw2wxyz


def is_close(co1: Coordinates, co2: Coordinates, pos_tol: float = 1e-3, rot_tol: float = 1e-2):
    p1 = co1.worldpos()
    p2 = co2.worldpos()
    pos_dist = np.linalg.norm(p1 - p2)
    if pos_dist > pos_tol:
        return False
    p1 = co1.worldrot()
    p2 = co2.worldrot()
    rot_dist = np.linalg.norm(p1 - p2)
    if rot_dist > rot_tol:
        return False

    return True


def get_skrobot_coords(body_id: int) -> Coordinates:
    trans, quat = pb.getBasePositionAndOrientation(body_id)
    mat = quaternion2matrix(xyzw2wxyz(quat))
    return Coordinates(trans, mat)
