"""Commannd line interface for FLFM."""

import io
from pathlib import Path
from typing import Literal

import fire

import flfm.io
import flfm.pytorch_io
import flfm.pytorch_restoration
import flfm.restoration
import flfm.util


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
    match backend:
        case "torch":
            backend_restoration = flfm.pytorch_restoration
            backend_io = flfm.pytorch_io
        case "jax":
            backend_restoration = flfm.restoration
            backend_io = flfm.io
        case _:
            raise ValueError(f"{backend} is not supported.")

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


if __name__ == "__main__":
    fire.Fire(main)
