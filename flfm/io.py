"""Image/data io operations."""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import tensorflow as tf
from jax.experimental import jax2tf
from PIL import Image

from flfm.restoration import compute_step_f


def open(filename: str | Path) -> jnp.ndarray:
    """Open a file and return it as a numpy array."""
    img = Image.open(filename)
    frames = []
    for i in range(img.n_frames):
        img.seek(i)
        frames.append(jnp.array(img))

    return jnp.stack(frames, axis=0)  # [n_frames, h, w]


def save(
    filename: str | Path,
    data: jnp.ndarray,  # [n_frames, h, w]
    format=None,
) -> None:
    """Save a numpy array to file. The file format is determined by the filename suffix when present."""
    img = Image.fromarray(np.array(data[0]))
    img.save(filename, format=format, save_all=True, append_images=[Image.fromarray(np.array(d)) for d in data[1:]])


def export_model(
    out_path: str | Path,
    num_steps: int,
    img_size: tuple[int, int, int],
    psf_size: tuple[int, int, int],
) -> None:
    """Unroll the Richardson-Lucy algorithm for a given number of steps and save it.

    Args:
        out_path: Path to save the model.
        num_steps: Number of steps to unroll.
        img_size: Size of the image tensor.
        psf_size: Size of the PSF tensor.

    Returns:
        None
    """

    def rl(img: jnp.ndarray, psf: jnp.ndarray) -> jnp.ndarray:
        psf_fft = jnp.fft.rfft2(psf)  # [k, n, n/2+1]
        psft_fft = jnp.fft.rfft2(jnp.flip(psf))  # [k, n, n/2+1]
        data = jnp.ones_like(psf) * 0.5  # [k, n, n]

        for _ in range(num_steps):
            data = compute_step_f(data, img, psf_fft, psft_fft)

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
