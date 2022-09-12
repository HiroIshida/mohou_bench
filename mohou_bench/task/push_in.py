import pybullet as pb
import pybullet_data

from mohou_bench.pybullet_utils import BoxConfig, create_box
from mohou_bench.teleop import KeyboardCommander

if __name__ == "__main__":
    pb.connect(pb.GUI)
    pb.configureDebugVisualizer(pb.COV_ENABLE_GUI, 0)
    pb.setGravity(0, 0, -10)
    pb.setAdditionalSearchPath(pybullet_data.getDataPath())  # used by loadURDF
    pb.loadURDF("plane.urdf")
    com = KeyboardCommander.create()

    box_id = create_box(
        BoxConfig(size=(0.04, 0.04, 0.04), rgba="pale_red"), pos=(0.5, 0.0), friction=0.3
    )

    width = 0.15
    depth = 0.15
    create_box(
        BoxConfig(size=(depth, 0.02, 0.02), rgba="gray"), pos=(0.7, -0.5 * width), fixed=True
    )
    create_box(
        BoxConfig(size=(depth, 0.02, 0.02), rgba="gray"), pos=(0.7, +0.5 * width), fixed=True
    )
    create_box(
        BoxConfig(size=(0.02, width, 0.02), rgba="gray"),
        pos=(0.7 + depth * 0.5 - 0.01, 0.0),
        fixed=True,
    )

    com.run()
