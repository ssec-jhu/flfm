import numpy as np
import pytest

import flfm.io
import flfm.restoration
import flfm.util
from flfm.backend import reload_backend
from flfm.settings import settings

DATA_DIR = flfm.util.find_package_location() / "tests" / "data" / "yale"
IMAGE_FILENAME = DATA_DIR / "light_field_image.tif"
PSF_FILENAME = DATA_DIR / "measured_psf.tif"
RECONSTRUCTION_FILENAME = DATA_DIR / ".." / "ssec" / "reconstruction.tif"


class TestRichardsonLucy:
    @pytest.mark.parametrize("backend", ("torch",))
    def test_2d_image(self, monkeypatch, backend):
        monkeypatch.setattr(settings, "BACKEND", backend)
        reload_backend(backend)

        image = flfm.io.open(IMAGE_FILENAME)
        psf = flfm.io.open(PSF_FILENAME)
        psf = psf / flfm.restoration.sum(psf)

        reconstruction = flfm.restoration.reconstruct(image, psf)
        assert reconstruction.shape == psf.shape

        cropped_recon = flfm.util.crop_and_apply_circle_mask(
            reconstruction,
            center=(settings.DEFAULT_CENTER_X, settings.DEFAULT_CENTER_Y),
            radius=settings.DEFAULT_RADIUS,
        )
        assert np.allclose(cropped_recon, flfm.io.open(RECONSTRUCTION_FILENAME))
