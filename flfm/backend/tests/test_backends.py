import io

import numpy as np
import pytest

import flfm.backend.jax
import flfm.backend.numpy
import flfm.backend.torch
import flfm.io
import flfm.restoration
import flfm.util
from flfm.backend import reload_backend
from flfm.settings import settings
from flfm.tests.conftest import arr_to_stream

BACKENDS = ("torch", "jax", "numpy")


class TestBackendReload:
    @pytest.mark.parametrize("backend", BACKENDS)
    def test_reload_backend(self, backend):
        reload_backend(backend)
        assert settings.BACKEND == backend

        match backend:
            case "jax":
                assert flfm.io.IO is flfm.backend.jax.JaxIO()
                assert flfm.io.open is flfm.backend.jax.JaxIO().open
                assert flfm.io.save is flfm.backend.jax.JaxIO().save
                assert flfm.restoration.Restoration is flfm.backend.jax.JaxRestoration()
                assert flfm.restoration.sum is flfm.backend.jax.JaxRestoration().sum
                assert flfm.restoration.export_model is flfm.backend.jax.JaxRestoration().export_model
            case "torch":
                assert flfm.io.IO is flfm.backend.torch.TorchIO()
                assert flfm.io.open is flfm.backend.torch.TorchIO().open
                assert flfm.io.save is flfm.backend.torch.TorchIO().save
                assert flfm.restoration.Restoration is flfm.backend.torch.TorchRestoration()
                assert flfm.restoration.sum is flfm.backend.torch.TorchRestoration().sum
                assert flfm.restoration.export_model is flfm.backend.torch.TorchRestoration().export_model
            case "numpy":
                assert flfm.io.IO is flfm.backend.numpy.NumpyIO()
                assert flfm.io.open is flfm.backend.numpy.NumpyIO().open
                assert flfm.io.save is flfm.backend.numpy.NumpyIO().save
                assert flfm.restoration.Restoration is flfm.backend.numpy.NumpyRestoration()
                assert flfm.restoration.sum is flfm.backend.numpy.NumpyRestoration().sum
                assert flfm.restoration.export_model is flfm.backend.numpy.NumpyRestoration().export_model
            case _:
                raise NotImplementedError

    def test_invalid_backend(self):
        """Test the backend validation function with an invalid backend."""
        with pytest.raises(NotImplementedError):
            reload_backend("invalid_backend")


class TestBackend:
    def test_singleton(self):
        assert flfm.backend.jax.JaxIO() is flfm.backend.jax.JaxIO()
        assert flfm.backend.torch.TorchIO() is flfm.backend.torch.TorchIO()

        assert flfm.backend.jax.JaxRestoration() is flfm.backend.jax.JaxRestoration()
        assert flfm.backend.torch.TorchRestoration() is flfm.backend.torch.TorchRestoration()

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_simple_integration(self, backend):
        """Test the backend integration."""

        # Explicitly import for sanity rather than using ``flfm.backend.reload_backend``.
        match backend:
            case "jax":
                io_backend = flfm.backend.jax.JaxIO()
                restoration_backend = flfm.backend.jax.JaxRestoration()
            case "torch":
                io_backend = flfm.backend.torch.TorchIO()
                restoration_backend = flfm.backend.torch.TorchRestoration()
            case "numpy":
                io_backend = flfm.backend.numpy.NumpyIO()
                restoration_backend = flfm.backend.numpy.NumpyRestoration()
            case _:
                raise NotImplementedError

        r = 5
        n, h, w = 4, 32, 32

        # setup a random image and psf and save them to a byte stream
        rand_img = np.ones([1, h, w], dtype=np.float32)
        rand_img_stream = arr_to_stream(rand_img)

        sim_psf = np.ones([n, h, w], dtype=np.float32) / (w * h)
        sim_psf_stream = arr_to_stream(sim_psf)

        # read them back in from the byte stream
        img = io_backend.open(rand_img_stream)
        psf = io_backend.open(sim_psf_stream)

        # reconstruct the image
        reconstructed = restoration_backend.richardson_lucy(img, psf, num_iter=1)

        # save the reconstructed image to a byte stream
        out_stream = io.BytesIO()
        io_backend.save(out_stream, reconstructed, format="TIFF")
        out_stream.seek(0)

        # read the reconstructed image back in from the byte stream
        out_img = io_backend.open(out_stream)

        # make a circle mask
        mask = flfm.util.make_circle_mask(r)
        out = flfm.util.crop_and_apply_circle_mask(sim_psf, (h // 2, w // 2), r)

        expected_area = np.ceil(np.pi * r**2)
        assert img.shape == (1, h, w)
        assert psf.shape == (n, h, w)
        assert reconstructed.shape == (n, h, w)
        assert out_img.shape == (n, h, w)
        assert expected_area == mask.sum()
        assert mask.shape == (2 * r, 2 * r)
        assert out.shape == (n, 2 * r, 2 * r)
