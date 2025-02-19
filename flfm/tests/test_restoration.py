import time

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


class TestPerformance:
    def test_3d_reconstruction(self):
        image = flfm.io.open(flfm.util.find_repo_location() / "data/yale/light_field_image.tif")
        psf = flfm.io.open(flfm.util.find_repo_location() / "data/yale/measured_psf.tif")
        psf /= jnp.sum(psf, axis=(1, 2), keepdims=True)
        t0 = time.time()
        recon = flfm.restoration.richardson_lucy(image, psf, wait=True)
        print(f"test_3d_reconstruction RT: {int((time.time() - t0) * 1e3)}ms")
        assert recon.shape == psf.shape

        truth = flfm.io.open(flfm.util.find_repo_location() / "data/ssec/reconstructed_image.tif")
        cropped = flfm.util.crop_and_apply_circle_mask(recon, center=CENTER, radius=RADIUS)
        assert np.isclose(truth, cropped).all()

    def test_3d_reconstruction_perf(self):
        image = flfm.io.open(flfm.util.find_repo_location() / "data/yale/light_field_image.tif")
        psf = flfm.io.open(flfm.util.find_repo_location() / "data/yale/measured_psf.tif")
        psf /= jnp.sum(psf, axis=(1, 2), keepdims=True)

        # Precompute the FFT of the PSFs
        psf_fft = jnp.fft.rfft2(psf, axes=(-2, -1))
        psft_fft = jnp.fft.rfft2(jnp.flip(psf, axis=(-2, -1)))

        initial_guess = jnp.ones_like(psf) * 0.5

        t0 = time.time()
        recon = flfm.restoration.richardson_lucy(
            image, psf=None, initial_guess=initial_guess, psf_fft=psf_fft, psft_fft=psft_fft, wait=True
        )
        print(f"test_3d_reconstruction_perf RT: {int((time.time() - t0) * 1e3)}ms")
        assert recon.shape == psf.shape

        truth = flfm.io.open(flfm.util.find_repo_location() / "data/ssec/reconstructed_image.tif")
        cropped = flfm.util.crop_and_apply_circle_mask(recon, center=CENTER, radius=RADIUS)
        assert np.isclose(truth, cropped).all()
