import multiprocessing
import shutil
from pathlib import Path

from dask.distributed import LocalCluster

import flfm.util
from flfm.settings import settings

# TODO: Refactor this to some __init__.py. See https://github.com/ssec-jhu/flfm/issues/146.
match settings.BACKEND:
    case "torch":
        import flfm.pytorch_io as flfm_io
        import flfm.pytorch_restoration as flfm_restoration
    case "jax":
        import flfm.io as flfm_io
        import flfm.restoration as flfm_restoration
    case _:
        raise NotImplementedError(f"Backend {settings.BACKEND} not implemented.")


def do_reconstruction(psf, image_filename, output_filename, recon_kwargs, crop_kwargs, write=False):
    """Do full reconstruction and save to file."""
    # TODO: Move this func to same place that wrappers live for #146 & #135.

    # Open light-field image.
    image = flfm_io.open(image_filename)
    # Reconstruct.
    reconstruction = flfm_restoration.richardson_lucy(image, psf, **recon_kwargs)
    # Crop.
    cropped_reconstruction = flfm.util.crop_and_apply_circle_mask(reconstruction, **crop_kwargs)
    # Save.
    if write:
        flfm_io.save(output_filename, cropped_reconstruction)
    return cropped_reconstruction


def batch_reconstruction(
    input_dir: str | Path,
    output_dir: str | Path,
    psf,
    n_workers: int | None = None,
    n_threads: int = 2,
    clobber=False,
    recon_kwargs=None,
    crop_kwargs=None,
    carry_on=False,
):
    """
    Batch parallel process multiple 3D reconstructions of input light-field images present in `input_dir`.

    :param input_dir: Input directory.
    :param output_dir: Output directory.
    :param psf: Loaded PSF image.
    :param n_workers: Numbers of parallel workers.
    :param n_threads: Number of threads per worker.
    :param clobber: Write over files in `output_dir`, otherwise raise if `output_dir` exists. Defaults to `False`.
    :param recon_kwargs: kwargs passed to `richardson_lucy()`.
    :param crop_kwargs: kwargs pass to `flfm.util.crop_and_apply_circle_mask()`.
    :param carry_on: Whether to continue processing from a previous attempt. Will only process input files not in
    `output_dir`.
    :return: List of processed filenames.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=clobber or carry_on)

    input_dir = Path(input_dir)
    input_filenames = flfm.util.find_files(input_dir)

    if carry_on:
        existing_output_files = [x.name for x in flfm.util.find_files(output_dir)]
        n_input_files = len(input_filenames)
        input_filenames = [x for x in input_filenames if x.name not in existing_output_files]
        print(f"Carrying on batch processing at {len(input_filenames)}/{n_input_files}.")

    recon_kwargs = recon_kwargs or dict(num_iter=settings.DEFAULT_RL_ITERS)
    crop_kwargs = crop_kwargs or dict(
        center=(settings.DEFAULT_CENTER_X, settings.DEFAULT_CENTER_Y), radius=settings.DEFAULT_RADIUS
    )

    cluster = LocalCluster(n_workers=n_workers or multiprocessing.cpu_count(), threads_per_worker=n_threads)
    client = cluster.get_client()

    futures = []
    for i, filename in enumerate(input_filenames):
        print(f"({i + 1}/{len(input_filenames)}): Processing '{filename}'...")
        output_filename = output_dir / filename
        future = client.submit(
            do_reconstruction,
            psf,
            filename,
            output_filename,
            recon_kwargs=recon_kwargs,
            crop_kwargs=crop_kwargs,
            write=True,
        )
        futures.append(future)

    client.gather(futures)
    return input_filenames


def mock_batch_data(image_filename: str | Path, output_dir: str | Path, n_copies: int):
    """This helper func can be used to create mock data sets/dirs for, e.g., perf testing."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_filename = Path(image_filename)

    def do_copy(input_filename, output_filename):
        shutil.copy(input_filename, output_filename)

    cluster = LocalCluster(n_workers=multiprocessing.cpu_count(), threads_per_worker=2)
    client = cluster.get_client()

    futures = []
    for i in range(n_copies):
        future = client.submit(
            do_copy,
            image_filename,
            output_dir / f"{image_filename.stem}_{i + 1}{image_filename.suffix}",
        )
        futures.append(future)

    client.gather(futures)
