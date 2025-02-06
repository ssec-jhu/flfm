"""Commannd line interface for FLFM."""

import io
from pathlib import Path

import fire

import flfm.io as flfm_io
import flfm.reconstruct as flfm_reconstruct
import flfm.util as flfm_utils


def main(
    img: Path | str | io.BytesIO,
    psf: Path | str | io.BytesIO,
    out: Path | str | io.BytesIO,
    lens_radius: int,
    num_iters: int = 10,
    lens_center: tuple[int, int] | None = None,
) -> None:
    """Run the command line interface.

    Args:
        img: Path to the input image file.
        psf: Path to the PSF file.
        out: Path to the output file.
        lens_radius: Radius of the lens mask to apply to the output
        num_iters: Number of iterations to run, default is 10.
        lens_center: Center of the lens to apply the circular mask to
    """
    img = flfm_io.open_tiff(img)
    psf = flfm_io.open_tiff(psf)
    lens_center = lens_center or (img.shape[-2] // 2, img.shape[-1] // 2)

    reconstructed = flfm_reconstruct.reconstruct(img, psf, num_iter=num_iters)
    cropped = flfm_utils.crop_and_apply_circle_mask(
        reconstructed,
        lens_center,
        lens_radius,
    )

    flfm_io.save_tiff(out, cropped)


if __name__ == "__main__":
    fire.Fire(main)
