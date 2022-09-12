import pybullet as pb

from mohou_bench.pybullet_utils import BoxConfig, CylinderConfig


def test_box_config() -> None:
    pb.connect(pb.DIRECT)
    config = BoxConfig((0.1, 0.1, 0.1), "pale_red")
    config.to_pybullet_object(pos=(0.2, 0.2))


def test_cylinder_config() -> None:
    pb.connect(pb.DIRECT)
    config = CylinderConfig(0.1, 0.2, "pale_red")
    config.to_pybullet_object(pos=(0.2, 0.2))
