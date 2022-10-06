import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import gdown


def get_cache_path() -> Path:
    p = Path("~/.cache/mohou_bench").expanduser()
    p.mkdir(exist_ok=True)
    return p


def get_fryingpan_urdf_path() -> Path:

    url = "https://drive.google.com/uc?id=1wtKEeOqKMg8h36LLqP64kzsgKUbtnDkA"

    urdf_path = get_cache_path() / "Chefmate_8_Frypan" / "object.urdf"

    if not urdf_path.exists():
        with TemporaryDirectory() as f:
            tar_path = Path(f) / "tmp.tar"
            gdown.download(url, str(tar_path), quiet=False)

            cmd = "cd {cache_path} && tar xf {tar_path}".format(
                cache_path=get_cache_path(), tar_path=tar_path
            )
            subprocess.run(cmd, shell=True)

    assert urdf_path.exists()
    return urdf_path
