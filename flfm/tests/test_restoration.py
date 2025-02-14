import jax.numpy as jnp
import numpy as np

import flfm.io
import flfm.restoration
import flfm.util

CENTER = [1000, 980]
RADIUS = 230


class TestRichardsonLucy:
    def test_2d_image(self):
        image = flfm.io.open(flfm.util.find_repo_location() / "data/yale/light_field_image.tif")
        psf = flfm.io.open(flfm.util.find_repo_location() / "data/yale/measured_psf.tif")
        recon = flfm.restoration.richardson_lucy(image, psf[21])
        assert recon.shape == image.shape

    def test_3d_reconstruction(self):
        image = flfm.io.open(flfm.util.find_repo_location() / "data/yale/light_field_image.tif")
        psf = flfm.io.open(flfm.util.find_repo_location() / "data/yale/measured_psf.tif")

        # Normalize PSF.
        psf /= jnp.sum(psf, axis=(1, 2), keepdims=True)

        recon = flfm.restoration.richardson_lucy(image, psf)
        assert recon.shape == psf.shape
        cropped = flfm.util.crop_and_apply_circle_mask(recon, center=CENTER, radius=RADIUS)
        truth = flfm.io.open(flfm.util.find_repo_location() / "data/yale/reconstructed_image.tif")
        assert cropped.shape == truth.shape
        assert np.isclose(truth, cropped).all()
