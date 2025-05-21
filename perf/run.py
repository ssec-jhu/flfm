import cProfile
import os
import subprocess
from pathlib import Path
from typing import Protocol

import fire
import memray

from flfm import cli


class Profiler:
    def __init__(self, fname: str|Path) -> None:
        self._fname = fname

    def __enter__(self) -> None:
        raise NotImplementedError("Profiler must be implemented in a subclass")

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        raise NotImplementedError("Profiler must be implemented in a subclass")


class CProfiler(Profiler):
    def __init__(self, fname: str|Path) -> None:
        super().__init__(fname)
        self._profiler = cProfile.Profile()

    def __enter__(self) -> None:
        self._profiler.enable()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._profiler.disable()
        self._profiler.dump_stats(self._fname)


class NsysProfiler(Profiler):
    def __init__(self, fname: str|Path) -> None:
        super().__init__(fname)
        self._profiler = None

    def __enter__(self) -> None:
        # docs here:
        # https://docs.nvidia.com/nsight-systems/UserGuide/index.html#cli-start-command-switch-options
        subprocess.run([
            "nsys", "start",
            "--force-overwrite=true",
            f"--output={self._fname}.nsys-rep",
            "--trace=cuda,osrt",
            "",
        ])
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        # docs here:
        # https://docs.nvidia.com/nsight-systems/UserGuide/index.html#cli-stop-command-switch-options
        subprocess.run(["nsys", "stop"])
        # docs here:
        # https://docs.nvidia.com/nsight-systems/UserGuide/index.html#cli-stats-command-switch-options
        subprocess.run([
            "nsys", "report",
            "--report=csv",
            "--output", str(self._fname),
            f"{self._fname}.nsys-rep"
        ])
        os.remove(f"{self._fname}.nsys-rep")



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


def resolve_profiler(profiler_name: str, output_fname:str|Path) -> Profiler:
    match profiler_name:
        case "cProfile":
            return CProfiler(output_fname)
        case "nsys":
            return NsysProfiler(output_fname)
        case "memray":
            return memray.Tracker(output_fname)
        case _:
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


    with resolve_profiler(profiler_name, out_path) as profiler:
        _ = restoration.richardson_lucy(
            img,
            psf,
            num_iter=10,
        )


if __name__=="__main__":
    fire.Fire(run)