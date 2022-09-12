import pybullet as pb
import pybullet_data
from skrobot.coordinates.math import rpy2quaternion, wxyz2xyzw

from mohou_bench.pybullet_utils import BoxConfig, create_box
from mohou_bench.teleop import KeyboardCommander

if __name__ == "__main__":
    pb.connect(pb.GUI)
    pb.configureDebugVisualizer(pb.COV_ENABLE_GUI, 0)
    pb.setGravity(0, 0, -10)
    pb.setAdditionalSearchPath(pybullet_data.getDataPath())  # used by loadURDF
    pb.loadURDF("plane.urdf")
    com = KeyboardCommander.create()

    box_id = create_box(BoxConfig(size=(0.04, 0.04, 0.04), rgba=(1, 0.7, 0.7, 1.0)), friction=0.3)

    q = rpy2quaternion((0, 0, 0))
    pb.resetBasePositionAndOrientation(box_id, (0.5, 0.0, 0.02), wxyz2xyzw(q))
    com.run()
