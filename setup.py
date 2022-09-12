from setuptools import setup

setup_requires = []

install_requires = ["gdown", "scikit-robot", "pybullet", "pynput"]

setup(
    name="mohou_bench",
    version="0.0.0",
    description="task set for behavioral cloning",
    author="Hirokazu Ishida",
    author_email="h-ishida@jsk.imi.i.u-tokyo.ac.jp",
    url="https://github.com/HiroIshida/mohou_task_public",
    long_description_content_type="text/markdown",
    license="MIT",
    install_requires=install_requires,
)
