"""FLFM observation reconstruction."""

from pathlib import Path
from typing import Optional

from numpy.typing import ArrayLike

import flfm.util  # noqa:  F401
from flfm.settings import settings

match settings.BACKEND:
    case "jax":
        from flfm.backend.jax import JaxRestoration

        Restoration = JaxRestoration()
        assert Restoration is JaxRestoration()
    case "torch":
        from flfm.backend.torch import TorchRestoration

        Restoration = TorchRestoration()
        assert Restoration is TorchRestoration()
    case _:
        raise NotImplementedError


sum = Restoration.sum
export_model = Restoration.export_model


def reconstruct(
    image: ArrayLike | str | Path,
    psf: ArrayLike | str | Path,
    normalize_psf: bool = False,
    output_filename: Optional[str | Path] = None,
    recon_kwargs: Optional[dict] = None,
    crop_kwargs: Optional[dict] = None,
) -> ArrayLike:
    import flfm.io

    if isinstance(image, (str, Path)):
        image = flfm.io.open(Path(image))

    if isinstance(psf, (str, Path)):
        psf = flfm.io.open(Path(psf))

    if normalize_psf:
        psf = psf / sum(psf)

    if recon_kwargs is None:
        recon_kwargs = {}

    reconstruction = Restoration.richardson_lucy(image, psf, **recon_kwargs)

    if isinstance(crop_kwargs, dict):
        reconstruction = flfm.util.crop_and_apply_circle_mask(reconstruction, **crop_kwargs)

    if output_filename:
        flfm.io.save(output_filename, reconstruction)

    return reconstruction
