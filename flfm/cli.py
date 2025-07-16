"""Commannd line interface for FLFM."""

import io
import warnings
from pathlib import Path
from typing import Literal, Optional

import fire

import flfm.util  # noqa:  F401
from flfm.backend import reload_backend
from flfm.settings import settings

from batch import batch_reconstruction

ERR_BACKEND_MSSG = "FLFM {backend} not found ❌"
BACKEND_SUCCESS = "FLFM {backend} loaded ✅"


def import_backend(backend: str) -> None:
    """Import the specified backend.

    Args:
        backend: The name of the backend to import.
    """
    try:
        reload_backend(backend)
    except ImportError:
        warnings.warn(ERR_BACKEND_MSSG.format(backend=backend), ImportWarning)
    else:
        print(BACKEND_SUCCESS.format(backend=backend))


def main(
    img: str | Path | io.BytesIO,
    psf: str | Path | io.BytesIO,
    out: str | Path | io.BytesIO,
    lens_radius: int,
    num_iters: int = 10,
    normalize_psf: bool = False,
    lens_center: Optional[tuple[int, int]] = None,
    backend: Literal["jax", "torch"] = settings.BACKEND,
) -> None:
    """Run the command line interface.

    Args:
        img: Path to the input image file.
        psf: Path to the PSF file.
        out: Path to the output file.
        lens_radius: Radius of the lens mask to apply to the output
        num_iters: Number of iterations to run, default is 10.
        normalize_psf: Whether to normalize the PSF. Default is False.
        lens_center: Center of the lens to apply the circular mask to
        backend: Whether to use JAX or Torch. Default is "torch".
    """
    import_backend(backend)  # Reimports ``flfm.io`` & ``flfm.restoration`` to switch backend.
    # Even with the above reimport, still directly import for better readability and to also stop linter complaints.
    import flfm.io
    import flfm.restoration

    img = flfm.io.open(img)
    psf = flfm.io.open(psf)

    if normalize_psf:
        psf = psf / flfm.restoration.sum(psf)

    reconstructed = flfm.restoration.reconstruct(img, psf, recon_kwargs=dict(num_iter=num_iters))

    lens_center = lens_center or (img.shape[-2] // 2, img.shape[-1] // 2)
    cropped = flfm.util.crop_and_apply_circle_mask(
        reconstructed,
        lens_center,
        lens_radius,
    )

    flfm.io.save(out, cropped)


def export(
    out: str | Path | io.BytesIO,
    n_steps: int,
    backend: Literal["jax", "torch"] = "torch",
    img_size: tuple[int, int, int] = (1, 2048, 2048),
    psf_size: tuple[int, int, int] = (41, 2048, 2048),
) -> None:
    """Export a model for use elsewhere.

    Args:
        out: Path to the output file.
        n_steps: Number of steps to unroll.
        backend: Whether to use JAX or pytorch. Default is "torch".
        img_size: Size of the image tensor, should be (1, h, w).
        psf_size: Size of the PSF tensor, should be (k, h, w).

    Returns:
        None
    """
    import_backend(backend)
    import flfm.restoration  # Stop linter/IDE from complaining.

    flfm.restoration.export_model(
        Path(out),
        int(n_steps),
        img_size=img_size,
        psf_size=psf_size,
    )


def batch(
        input_dir: str | Path,
        output_dir: str | Path,
        psf_filename: str | Path,
        normalize_psf: bool = True,
        n_workers: Optional[int] = None,
        n_threads: int = 2,
        clobber: bool = False,
        recon_kwargs: Optional[dict] = None,
        crop_kwargs: Optional[dict] = None,
        carry_on: bool = False
)-> list[Path]:
    """Batch parallel process multiple 3D reconstructions of input light-field images present in `input_dir`.

        Args:
            input_dir: Input directory.
            output_dir: Output directory.
            psf_filename: PSF filename.
            normalize_psf: Whether to normalize PSF before reconstruction. Defaults to True.
            n_workers: Numbers of parallel workers.  Defaults to None.
            n_threads: Number of threads per worker.  Defaults to 2.
            clobber: Write over files in `output_dir`, otherwise raise if `output_dir` exists.
                Defaults to `False`.
            recon_kwargs: kwargs passed to `richardson_lucy()`.  Defaults to None.
            crop_kwargs: kwargs pass to `flfm.util.crop_and_apply_circle_mask()`. Defaults to None.
            carry_on: Whether to continue processing from a previous attempt. Will only process input
                files not in `output_dir`.  Defaults to False.

        Returns:
           A comma seperated string of processed filenames.
        """
    processed_filenames = batch_reconstruction(
        input_dir,
        output_dir,
        psf_filename,
        normalize_psf=normalize_psf,
        n_workers=n_workers,
        n_threads=n_threads,
        clobber=clobber,
        recon_kwargs=recon_kwargs,
        crop_kwargs=crop_kwargs,
        carry_on=carry_on,
    )
    return ",".join(processed_filenames)


if __name__ == "__main__":
    fire.Fire(
        {
            "main": main,
            "batch": batch,
            "export": export,
        }
    )
