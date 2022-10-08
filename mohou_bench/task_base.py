from typing import Dict, List

import numpy as np
import pybullet as pb
from skrobot.coordinates import Coordinates
from skrobot.coordinates.math import (
    quaternion2matrix,
    rpy2quaternion,
    wxyz2xyzw,
    xyzw2wxyz,
)

from mohou_bench.asset import BulletObject


class Task:
    object_table: Dict[str, BulletObject]

    def __init__(self, objects: List[BulletObject]):
        table = {}
        for obj in objects:
            table[obj.name] = obj
        self.object_table = table

    def update_world(self):
        for key, obj in self.object_table.items():
            point = obj.pose[:3]
            rpy = obj.pose[-3:]
            self.set_pose(key, point, rpy)

    def set_pose(
        self,
        body_name: str,
        point: np.ndarray,
        rpy: np.ndarray,
    ):
        body_id = self.object_table[body_name].bullet_id
        q = rpy2quaternion(rpy)
        pb.resetBasePositionAndOrientation(body_id, point, wxyz2xyzw(q))
        pb.resetBaseVelocity(
            body_id,
            linearVelocity=(0.0, 0.0, 0.0),
            angularVelocity=(0.0, 0.0, 0.0),
        )

    def get_skrobot_coords(self, body_name: str) -> Coordinates:
        # NOTE quat is xyzw order
        body_id = self.object_table[body_name].bullet_id
        trans, quat = pb.getBasePositionAndOrientation(body_id)
        mat = quaternion2matrix(xyzw2wxyz(quat))
        return Coordinates(trans, mat)
