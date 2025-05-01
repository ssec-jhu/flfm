"""Image/data io operations."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from flfm.pytorch_restoration import compute_step_f


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


def export_model(
    out_path: str | Path,
    num_steps: int,
    img_size: tuple[int, int, int] = (1, 2048, 2048),
    psf_size: tuple[int, int, int] = (41, 2048, 2048),
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

    def rl(img: torch.Tensor, psf: torch.Tensor) -> torch.Tensor:
        psf_fft = torch.fft.rfft2(psf)  # [k, n, n/2+1]
        psft_fft = torch.fft.rfft2(torch.flip(psf))  # [k, n, n/2+1]
        data = torch.ones_like(psf) * 0.5  # [k, n, n]

        for _ in range(num_steps):
            data = compute_step_f(data, img, psf_fft, psft_fft)

        return data

    jitted_fn = torch.jit.script(
        rl,
        example_inputs=(
            torch.zeros(*img_size, dtype=torch.float32),
            torch.zeros(*psf_size, dtype=torch.float32),
        ),
    )

    torch.jit.save(jitted_fn, out_path)
