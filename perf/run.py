import cProfile
import subprocess
from pathlib import Path
from typing import Protocol

import fire

from flfm import cli



class Profiler(Protocol):
    def enable(self) -> None:
        ...
    def disable(self) -> None:
        ...
    def dump_stats(self, filename: str|Path) -> None:
        ...


class NsysProfiler(Profiler):
    def __init__(self, fname: str|Path) -> None:
        self._fname = fname

    def enable(self) -> None:
        # docs here:
        # https://docs.nvidia.com/nsight-systems/UserGuide/index.html#cli-start-command-switch-options
        subprocess.run([
            "nsys", "start",
            "--force-overwrite=true",
            f"--output={self._fname}"
            "--trace=cuda,osrt",
            "",
        ])

    def disable(self) -> None:
        # docs here:
        # https://docs.nvidia.com/nsight-systems/UserGuide/index.html#cli-stop-command-switch-options
        subprocess.run(["nsys", "stop"])

    def dump_stats(self, filename: str|Path) -> None:
        # docs here:
        # https://docs.nvidia.com/nsight-systems/UserGuide/index.html#cli-stats-command-switch-options
        subprocess.run(["nsys", "report", "--report=csv", "--output", str(filename)])


def resolve_profiler(profiler_name: str) -> Profiler:
    if profiler_name == "cProfile":
        return cProfile.Profile()
    elif profiler_name == "nsys":
        return None
    else:
        raise ValueError(f"Unknown profiler: {profiler_name}")


def run(
    backend_name:str,
    profiler_name:str,
    out_path:Path,
) -> None:
    base_path = Path("flfm/tests/data/yale")
    img_path =  base_path / "light_field_image.tif"
    psf_path = base_path / "measured_psf.tif"

    restoration, io = cli._validate_backend(backend_name)
    img = io.open(img_path)
    psf = io.open(psf_path)

    profiler = resolve_profiler(profiler_name)
    profiler.enable()
    _ = restoration.richardson_lucy(
        img,
        psf,
        num_iter=10,
    )
    profiler.disable()
    profiler.dump_stats(out_path)


if __name__=="__main__":
    fire.Fire(run)