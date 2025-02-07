"""Commannd line interface for FLFM."""

import io
from pathlib import Path

import fire

import flfm.io
import flfm.restoration
import flfm.util


def run(
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
    img = flfm.io.open(img)
    psf = flfm.io.open(psf)
    lens_center = lens_center or (img.shape[-2] // 2, img.shape[-1] // 2)

    reconstructed = flfm.restoration.richardson_lucy(img, psf, num_iter=num_iters)
    cropped = flfm.util.crop_and_apply_circle_mask(
        reconstructed,
        lens_center,
        lens_radius,
    )

    flfm.io.save(out, cropped)


def export(out_path: str):
    """Export the reconstruction function to a TensorFlow SavedModel."""
    flfm.restoration.export_to_tf(out_path)


if __name__ == "__main__":
    fire.Fire({"run": run, "export": export})
