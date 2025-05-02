"""Commannd line interface for FLFM."""

import io
from pathlib import Path
from types import ModuleType
from typing import Literal

import fire

import flfm.io
import flfm.pytorch_io
import flfm.pytorch_restoration
import flfm.restoration
import flfm.util


def _validate_backend(backend: Literal["jax", "torch"]) -> tuple[ModuleType, ModuleType]:
    """Validate the backend and return the appropriate modules."""
    restoration, io = None, None
    match backend:
        case "torch":
            return flfm.pytorch_restoration, flfm.pytorch_io
        case "jax":
            return flfm.restoration, flfm.io
        case _:
            raise ValueError(f"{backend} is not supported.")

    return restoration, io


def main(
    img: Path | str | io.BytesIO,
    psf: Path | str | io.BytesIO,
    out: Path | str | io.BytesIO,
    lens_radius: int,
    num_iters: int = 10,
    lens_center: tuple[int, int] | None = None,
    backend: Literal["jax", "torch"] = "torch",
) -> None:
    """Run the command line interface.

    Args:
        img: Path to the input image file.
        psf: Path to the PSF file.
        out: Path to the output file.
        lens_radius: Radius of the lens mask to apply to the output
        num_iters: Number of iterations to run, default is 10.
        lens_center: Center of the lens to apply the circular mask to
        backend: Whether to use JAX or Torch. Default is "torch".
    """
    backend_restoration, backend_io = _validate_backend(backend)

    img = backend_io.open(img)
    psf = backend_io.open(psf)

    reconstructed = backend_restoration.richardson_lucy(img, psf, num_iter=num_iters)

    lens_center = lens_center or (img.shape[-2] // 2, img.shape[-1] // 2)
    cropped = flfm.util.crop_and_apply_circle_mask(
        reconstructed,
        lens_center,
        lens_radius,
    )

    flfm.io.save(out, cropped)


def export(
    out: Path | str | io.BytesIO,
    n_steps: int,
    img_size: tuple[int, int, int] = (1, 2048, 2048),
    psf_size: tuple[int, int, int] = (41, 2048, 2048),
    backend: Literal["jax", "torch"] = "torch",
) -> None:
    """Export a model for use elsewhere.

    Args:
        out: Path to the output file.
        n_steps: Number of steps to unroll.
        img_size: Size of the image tensor, should be (1, h, w).
        psf_size: Size of the PSF tensor, should be (k, h, w).
        backend: Whether to use JAX or pytorch. Default is "torch".

    Returns:
        None
    """
    _, backend_io = _validate_backend(backend)

    backend_io.export_model(
        out,
        n_steps,
        img_size=img_size,
        psf_size=psf_size,
    )


if __name__ == "__main__":
    fire.Fire(
        {
            "main": main,
            "export": export,
        }
    )
