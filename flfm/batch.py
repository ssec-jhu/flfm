import multiprocessing
from functools import partial
from pathlib import Path

from dask.distributed import LocalCluster

import flfm.util
from flfm.settings import settings

# TODO: Refactor this to some __init__.py. See https://github.com/ssec-jhu/flfm/issues/146.
match settings.BACKEND:
    case "torch":
        import torch

        psf_sum = partial(torch.sum, dim=(1, 2), keepdim=True)
        import flfm.pytorch_io as flfm_io
        import flfm.pytorch_restoration as flfm_restoration
    case "jax":
        import jax.numpy as jnp

        psf_sum = partial(jnp.sum, axis=(1, 2), keepdims=True)
        import flfm.io as flfm_io
        import flfm.restoration as flfm_restoration
    case _:
        raise NotImplementedError(f"Backend {settings.BACKEND} not implemented.")


def do_reconstruction(
    psf_filename, image_filename, output_filename, normalize_psf, recon_kwargs, crop_kwargs, write=False
):
    """Do full reconstruction and save to file."""
    # TODO: Move this func to same place that wrappers live for #146 & #135.

    # Open PSF image and optionally normalize.
    psf = flfm_io.open(psf_filename)
    if normalize_psf:
        psf = psf / psf_sum(psf)

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
    psf_filename: str | Path,
    normalize_psf: bool = True,
    n_workers: int | None = None,
    n_threads: int = 2,
    clobber: bool = False,
    recon_kwargs: dict | None = None,
    crop_kwargs: dict | None = None,
    carry_on: bool = False,
):
    """Batch parallel process multiple 3D reconstructions of input light-field images present in `input_dir`.

    Args:
        input_dir (:obj:`str` | :obj: `Path`): Input directory.
        output_dir (:obj:`str` | :obj: `Path`): Output directory.
        psf_filename (:obj:`str` | :obj: `Path`): PSF filename.
        normalize_psf (bool, optional): Whether to normalize PSF before reconstruction. Defaults to True.
        n_workers (int, optional): Numbers of parallel workers.  Defaults to None.
        n_threads (int, optional): Number of threads per worker.  Defaults to 2.
        clobber (bool, optional): Write over files in `output_dir`, otherwise raise if `output_dir` exists.
            Defaults to `False`.
        recon_kwargs (:obj:`dict`, optional): kwargs passed to `richardson_lucy()`.  Defaults to None.
        crop_kwargs (:obj:`dict`, optional):: kwargs pass to `flfm.util.crop_and_apply_circle_mask()`. Defaults to None.
        carry_on (bool, optional): Whether to continue processing from a previous attempt. Will only process input
            files not in `output_dir`.  Defaults to False.

    Returns:
       list[Path]: List of processed filenames.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=clobber or carry_on)

    input_dir = Path(input_dir)
    input_filenames = flfm.util.find_files(input_dir)

    psf_filename = Path(psf_filename)

    if carry_on:
        existing_output_files = [x.name for x in flfm.util.find_files(output_dir)]
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
            output_filename = output_dir / filename
            future = client.submit(
                do_reconstruction,
                psf_filename,
                filename,
                output_filename,
                normalize_psf=normalize_psf,
                recon_kwargs=recon_kwargs,
                crop_kwargs=crop_kwargs,
                write=True,
            )
            futures.append(future)

        client.gather(futures)
    return input_filenames
