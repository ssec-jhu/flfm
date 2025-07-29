"""Batch processing of FLFM reconstructions."""

import multiprocessing
from pathlib import Path
from typing import Optional

from dask.distributed import LocalCluster

import flfm.util
from flfm.settings import settings


def batch_reconstruction(
    input_dir: str | Path,
    output_dir: str | Path,
    psf_filename: str | Path,
    filename_pattern: str = "*",
    normalize_psf: bool = True,
    n_workers: Optional[int] = None,
    n_threads: int = 2,
    clobber: bool = False,
    recon_kwargs: Optional[dict] = None,
    crop_kwargs: Optional[dict] = None,
    carry_on: bool = False,
) -> list[Path]:
    """Batch parallel process multiple 3D reconstructions of input light-field images present in `input_dir`.

    Args:
        input_dir: Input directory.
        output_dir: Output directory.
        psf_filename: PSF filename.
        filename_pattern: input filename glob pattern. Defaults to "*".
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
       A list of processed filenames.
    """
    import flfm.restoration

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=clobber or carry_on)

    input_dir = Path(input_dir)
    input_filenames = list(input_dir.glob(filename_pattern))

    if not input_filenames:
        raise FileNotFoundError(f"No input files found in {input_dir}")

    psf_filename = Path(psf_filename)

    if carry_on:
        existing_output_files = [x.name for x in output_dir.glob(filename_pattern)]
        n_input_files = len(input_filenames)
        input_filenames = [x for x in input_filenames if x.name not in existing_output_files]
        print(f"Carrying on batch processing at {len(input_filenames)}/{n_input_files}.")

    recon_kwargs = recon_kwargs or dict(num_iter=settings.DEFAULT_RL_ITERS)
    crop_kwargs = crop_kwargs or dict(
        center=(settings.DEFAULT_CENTER_X, settings.DEFAULT_CENTER_Y), radius=settings.DEFAULT_RADIUS
    )

    with LocalCluster(n_workers=n_workers or multiprocessing.cpu_count(), threads_per_worker=n_threads) as cluster:
        client = cluster.get_client()

        futures = []
        for i, filename in enumerate(input_filenames):
            print(f"({i + 1}/{len(input_filenames)}): Processing '{filename}'...")
            output_filename = output_dir / filename.name
            future = client.submit(
                flfm.restoration.reconstruct,
                psf_filename,
                filename,
                output_filename=output_filename,
                normalize_psf=normalize_psf,
                recon_kwargs=recon_kwargs,
                crop_kwargs=crop_kwargs,
            )
            futures.append(future)

        client.gather(futures)
    return input_filenames
