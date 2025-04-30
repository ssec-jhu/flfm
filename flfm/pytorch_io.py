"""Image/data io operations."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image


def open(filename: str | Path) -> torch.Tensor:
    """Open a file and return it as a numpy array."""
    img = Image.open(filename)
    frames = []
    for i in range(img.n_frames):
        img.seek(i)
        frames.append(torch.from_numpy(np.array(img).astype(np.float32)))

    return torch.stack(frames, dim=0)  # [n_frames, h, w]


def save(
    filename: str | Path,
    data: torch.Tensor,  # [n_frames, h, w]
    format=None,
) -> None:
    """Save a numpy array to file. The file format is determined by the filename suffix when present."""
    img = Image.fromarray(np.array(data[0].numpy()))
    img.save(filename, format=format, save_all=True, append_images=[Image.fromarray(np.array(d)) for d in data[1:]])
