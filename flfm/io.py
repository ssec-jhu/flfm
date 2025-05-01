"""Image/data io operations."""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
from PIL import Image


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
    num_steps: int,
    out_path: str | Path,
    img_size: tuple[int, int, int] = (),
    psf_size: tuple[int, int, int] = (),
) -> None:
    """Unroll the Richardson-Lucy algorithm for a given number of steps amd save it."""
