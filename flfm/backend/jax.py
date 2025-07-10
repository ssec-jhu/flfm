"""FLFM observation reconstruction using JAX."""

from pathlib import Path
from typing import Any, Sequence

import jax
import numpy as np
from numpy.typing import ArrayLike
from PIL import Image

from flfm.backend.base import BaseIO, BaseRestoration


def rl_step(
    data: jax.numpy.ndarray,  # [k, n, n]
    image: jax.numpy.ndarray,  # [1, n, n]
    PSF_fft: jax.numpy.ndarray,  # [k, n, n/2+1]
    PSFt_fft: jax.numpy.ndarray,  # [k, n, n/2+1]
) -> jax.numpy.ndarray:
    """Single step of the multiplicative Richardson-Lucy deconvolution algorithm."""
    # NOTE: This could use ``cls.rfft2``, however, we don't so that we can keep this func static.
    denominator = jax.numpy.fft.irfft2(PSF_fft * jax.numpy.fft.rfft2(data)).sum(axis=0, keepdims=True)  # [1, n, n]
    img_err = image / denominator
    return data * jax.numpy.fft.fftshift(
        jax.numpy.fft.irfft2(jax.numpy.fft.rfft2(img_err) * PSFt_fft), axes=(-2, -1)
    )  # [k, n, n]


class JaxRestoration(BaseRestoration):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.compiled_rl_step = jax.jit(rl_step)

    def compute_step(self, *args, **kwargs):
        """Wrapper for calling ``self.compiled_func()``. Override where needed."""
        return self.compiled_rl_step(*args, **kwargs).block_until_ready()

    @staticmethod
    def rfft2(a: ArrayLike, *args, axis: Sequence[int] = (-2, -1), **kwargs):
        return jax.numpy.fft.rfft2(a, *args, axes=axis, **kwargs)

    @staticmethod
    def ones_like(a: ArrayLike, *args, **kwargs) -> ArrayLike:
        return jax.numpy.ones_like(a, *args, **kwargs)

    @staticmethod
    def flip(a: ArrayLike, *args, axis: Sequence[int] = None, **kwargs) -> ArrayLike:
        return jax.numpy.flip(a, *args, axis=axis, **kwargs)

    @staticmethod
    def sum(a: ArrayLike, *args, axis: Sequence[int] = (1, 2), keepdims: bool = True, **kwargs):
        return jax.numpy.sum(a, axis=axis, keepdims=keepdims)

    @staticmethod
    def to_device(data: ArrayLike, device: str | Any = None) -> ArrayLike:
        # Jax, to some degree, handles this automatically when jitting. However, we could be explicit and implement
        # ``jax.Array.to_device()``, however, we can just leave this as a NoOp for now.
        return data

    @staticmethod
    def export_model(
        out_path: str | Path,
        num_steps: int,
        img_size: tuple[int, int, int],
        psf_size: tuple[int, int, int],
    ) -> None:
        import tensorflow as tf
        from jax.experimental import jax2tf

        def rl(img, psf: jax.numpy.ndarray) -> jax.numpy.ndarray:
            psf_fft = jax.numpy.fft.rfft2(psf)  # [k, n, n/2+1]
            psft_fft = jax.numpy.fft.rfft2(jax.numpy.flip(psf))  # [k, n, n/2+1]
            data = jax.numpy.ones_like(psf) * 0.5  # [k, n, n]

            for _ in range(num_steps):
                data = rl_step(data, img, psf_fft, psft_fft)

            return data

        serialize_gpu = len(tf.config.list_physical_devices("GPU")) > 0
        exported_f = tf.Module()
        exported_f.f = tf.function(
            jax2tf.convert(
                rl,
                with_gradient=False,
                native_serialization_platforms=["cpu"] + serialize_gpu * ["cuda"],
            ),
            autograph=False,
            input_signature=[
                tf.TensorSpec(shape=img_size, dtype=tf.float32, name="image"),  # [1, n, n]
                tf.TensorSpec(shape=psf_size, dtype=tf.float32, name="psf"),  # [k, n, n]
            ],
        )

        exported_f.f(
            tf.ones(img_size, dtype=tf.float32, name="image"),
            tf.ones(psf_size, dtype=tf.float32, name="psf"),
        )

        tf.saved_model.save(exported_f, out_path)


class JaxIO(BaseIO):
    @staticmethod
    def open(filename: str | Path) -> ArrayLike:
        """Open a file and return it as a numpy array."""
        img = Image.open(filename)
        frames = []
        for i in range(img.n_frames):
            img.seek(i)
            frames.append(jax.numpy.array(img))

        return jax.numpy.stack(frames, axis=0)  # [n_frames, h, w]

    @staticmethod
    def save(
        filename: str | Path,
        data: ArrayLike,  # [n_frames, h, w]
        format=None,
    ) -> None:
        """Save a numpy array to file. The file format is determined by the filename suffix when present."""
        img = Image.fromarray(np.array(data[0]))
        img.save(filename, format=format, save_all=True, append_images=[Image.fromarray(np.array(d)) for d in data[1:]])
