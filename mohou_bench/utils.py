import numpy as np
from skrobot.coordinates import Coordinates


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
