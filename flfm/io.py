"""Image/data io operations."""

import jax.numpy as jnp
import numpy as np
from PIL import Image


def open_tiff(filename: str) -> jnp.ndarray:
    """Open a tiff file and return it as a numpy array."""
    img = Image.open(filename)
    frames = []
    for i in range(img.n_frames):
        img.seek(i)
        frames.append(jnp.array(img))

    return jnp.stack(frames, axis=0)  # [n_frames, h, w]


def save_tiff(
    filename: str,
    data: jnp.ndarray,  # [n_frames, h, w]
) -> None:
    """Save a numpy array as a tiff file."""
    img = Image.fromarray(np.array(data[0]))
    img.save(filename, format="TIFF", save_all=True, append_images=[Image.fromarray(np.array(d)) for d in data[1:]])
