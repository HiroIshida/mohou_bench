from dataclasses import dataclass
from enum import Enum

import numpy as np
import pybullet as pb
from skrobot.coordinates import Coordinates
from skrobot.coordinates.geo import orient_coords_to_axis
from skrobot.coordinates.math import rotation_matrix_from_axis


@dataclass
class Camera:
    """Camera
    Most of the functions in this class
    are took from https://github.com/kosuke55/hanging_points_cnn
    Copyright (c)  2021 Kosuke Takeuchi
    """

    coords: Coordinates
    resolution: int

    class CameraPosition(Enum):
        front = ((1.9, 0, 0.7), (0.5, 0, 0.3))
        frontclose = ((1.2, 0, 0.6), (0.6, 0, 0.2))
        fronttop = ((1.3, 0, 1.2), (0.5, 0, 0.3))
        lefttop = ((0.7, 0.4, 1.5), (0.5, 0.0, 0.3))
        righttop = ((0.5, -0.9, 0.6), (0.5, -0.3, 0.3))

    @classmethod
    def create(cls, camera_pos: CameraPosition, n_pixel=224) -> "Camera":
        pos, lookat = camera_pos.value
        camera = cls(Coordinates(pos), n_pixel)
        camera.look_at(np.array(lookat), horizontal=True)
        return camera

    def draw_camera_pos(self) -> None:
        pb.removeAllUserDebugItems()
        start = self.coords.worldpos()
        end_x = start + self.coords.rotate_vector([0.1, 0, 0])
        pb.addUserDebugLine(start, end_x, [1, 0, 0], 3)
        end_y = start + self.coords.rotate_vector([0, 0.1, 0])
        pb.addUserDebugLine(start, end_y, [0, 1, 0], 3)
        end_z = start + self.coords.rotate_vector([0, 0, 0.1])
        pb.addUserDebugLine(start, end_z, [0, 0, 1], 3)

    def look_at(self, p: np.ndarray, horizontal=False) -> None:
        if np.all(p == self.coords.worldpos()):
            return
        z = p - self.coords.worldpos()
        orient_coords_to_axis(self.coords, z)
        if horizontal:
            self.coords.newcoords(
                Coordinates(
                    pos=self.coords.worldpos(),
                    rot=rotation_matrix_from_axis(z, [0, 0, -1], axes="zy"),
                )
            )

    def render(self) -> np.ndarray:
        target = self.coords.worldpos() + self.coords.rotate_vector([0, 0, 1.0])
        up = self.coords.rotate_vector([0, -1.0, 0])
        vm = pb.computeViewMatrix(self.coords.worldpos(), target, up)
        fov, aspect, near, far = 45.0, 1.0, 0.01, 5.1
        pm = pb.computeProjectionMatrixFOV(fov, aspect, near, far)
        _, _, rgba, depth, _ = pb.getCameraImage(
            self.resolution,
            self.resolution,
            vm,
            pm,
            renderer=pb.ER_TINY_RENDERER,
        )
        rgb = rgba[:, :, :3]
        return rgb
