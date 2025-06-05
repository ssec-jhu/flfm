import cProfile
import os
import subprocess
from pathlib import Path

import fire
import memray
import torch

from flfm import cli


class Profiler:
    def __init__(self, fname: str | Path) -> None:
        self._fname = fname

    def __enter__(self) -> None:
        raise NotImplementedError("Profiler must be implemented in a subclass")

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        raise NotImplementedError("Profiler must be implemented in a subclass")


class CProfiler(Profiler):
    def __init__(self, fname: str | Path) -> None:
        super().__init__(fname)
        self._profiler = cProfile.Profile()

    def __enter__(self) -> None:
        self._profiler.enable()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._profiler.disable()
        self._profiler.dump_stats(self._fname)


class NsysProfiler(Profiler):
    def __init__(self, fname: str | Path) -> None:
        super().__init__(fname)
        self._profiler = None

    def __enter__(self) -> None:
        # docs here:

        print("writing to", self._fname + ".nsys-rep")
        # https://docs.nvidia.com/nsight-systems/UserGuide/index.html#cli-start-command-switch-options
        subprocess.run(
            [
                "nsys",
                "start",
                "--force-overwrite=true",
                f"--output={self._fname}.nsys-rep",
            ]
        )
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        # docs here:
        # https://docs.nvidia.com/nsight-systems/UserGuide/index.html#cli-stop-command-switch-options
        if subprocess.run(["nsys", "stop"]).returncode != 0:
            raise RuntimeError("Nsight Systems profiler did not stop correctly.")

        # docs here:
        # https://docs.nvidia.com/nsight-systems/UserGuide/index.html#cli-stats-command-switch-options
        if (
            subprocess.run(
                [
                    "nsys",
                    "stats",
                    "--format=csv",
                    "--force-export=true",
                    "--output",
                    str(self._fname) + ".csv",
                    f"{self._fname}.nsys-rep",
                ]
            ).returncode
            != 0
        ):
            raise RuntimeError("Nsight Systems profiler did not generate stats correctly.")
        os.remove(f"{self._fname}.nsys-rep")


class TorchProfiler(Profiler):
    def __init__(self, fname: str | Path) -> None:
        super().__init__(fname)
        self._profiler = None

    def __enter__(self) -> None:
        self._profiler = torch.profiler.profile(
            activities=[
                # torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            on_trace_ready=torch.profiler.tensorboard_trace_handler(str(self._fname)),
            record_shapes=True,
            with_stack=True,
            profile_memory=True,
        )
        self._profiler.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._profiler.__exit__(exc_type, exc_value, traceback)
        # The CPU/CUDA  timings are embedded in the string in the caall below:
        print(self._profiler.key_averages().table())
        # The are the last part od the string and look like the following:
        # Self CPU time total: 919.879ms
        # Self CUDA time total: 760.885ms
        # self._profiler.export_chrome_trace(f"{self._fname}.json")


def resolve_profiler(profiler_name: str, output_fname: str | Path) -> Profiler:
    match profiler_name:
        case "cProfile":
            return CProfiler(output_fname)
        case "nsys":
            return NsysProfiler(output_fname)
        case "memray":
            return memray.Tracker(output_fname)
        case "torch":
            return TorchProfiler(output_fname)
        case _:
            raise ValueError(f"Unknown profiler: {profiler_name}")


def run(
    backend_name: str,
    profiler_name: str,
    out_path: Path,
) -> None:
    base_path = Path("flfm/tests/data/yale")
    img_path = base_path / "light_field_image.tif"
    psf_path = base_path / "measured_psf.tif"

    restoration, io = cli._validate_backend(backend_name)
    img = io.open(img_path)
    psf = io.open(psf_path)

    with resolve_profiler(profiler_name, out_path):
        _ = restoration.richardson_lucy(
            img,
            psf,
            num_iter=10,
        )
        if backend_name == "torch":
            torch.cuda.synchronize()


if __name__ == "__main__":
    fire.Fire(run)
