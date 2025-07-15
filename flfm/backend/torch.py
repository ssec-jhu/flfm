"""FLFM observation reconstruction using PyTorch."""

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from numpy.typing import ArrayLike
from PIL import Image

from flfm.backend.base import BaseIO, BaseRestoration


def rl_step(
    data: torch.Tensor,  # [k, n, n]
    image: torch.Tensor,  # [1, n, n]
    PSF_fft: torch.Tensor,  # [k, n, n/2+1]
    PSFt_fft: torch.Tensor,  # [k, n, n/2+1]
) -> torch.Tensor:
    """Single step of the multiplicative Richardson-Lucy deconvolution algorithm.

    Args:
        data: The current estimate of the deconvolved image.
        image: The observed image.
        PSF_fft: The FFT of the point spread function.
        PSFt_fft: The FFT of the time-reversed point spread function.

    Returns:
        The next estimate of the deconvolved image.
    """
    # NOTE: This could use ``TorchRestoration.rfft2``, however, we don't so that we can keep this func
    # independent/static.
    denominator = torch.fft.irfft2(PSF_fft * torch.fft.rfft2(data), dim=(-2, -1)).sum(dim=0, keepdim=True)  # [1, n, n]
    img_err = image / denominator
    return data * torch.fft.fftshift(torch.fft.irfft2(torch.fft.rfft2(img_err) * PSFt_fft), dim=(-2, -1))  # [k, n, n]


class TorchRestoration(BaseRestoration):
    """PyTorch implementation of the restoration backend."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.compiled_rl_step = torch.jit.script(rl_step)

    @staticmethod
    def rfft2(a: ArrayLike, *args, axis: Sequence[int] = (-2, -1), **kwargs) -> ArrayLike:
        """Compute the 2D real-to-complex FFT."""
        return torch.fft.rfft2(a, *args, dim=axis, **kwargs)

    @staticmethod
    def ones_like(a: ArrayLike, *args, **kwargs) -> ArrayLike:
        """Return an array of ones with the same shape and type as a given array."""
        return torch.ones_like(a, *args, **kwargs)

    @staticmethod
    def flip(a: ArrayLike, *args, axis: Any = None, **kwargs) -> ArrayLike:
        """Reverse the order of elements in an array along the given axes."""
        return torch.flip(a, *args, dims=axis, **kwargs)

    @staticmethod
    def to_device(data: ArrayLike, device: str | Any = None) -> ArrayLike:
        """Move data to a device."""
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        return data.to(device)

    @staticmethod
    def sum(a: ArrayLike, *args, axis: Sequence[int] = (1, 2), keepdims: bool = True, **kwargs):
        """Sum of array elements over a given axis."""
        return torch.sum(a, dim=axis, keepdim=keepdims)

    @staticmethod
    def export_model(
        out_path: str | Path,
        num_steps: int,
        img_size: tuple[int, int, int],
        psf_size: tuple[int, int, int],
    ) -> None:
        """Export a model for use elsewhere."""

        class RL(torch.nn.Module):
            """Richardson-Lucy deconvolution module."""

            n_iter: torch.jit.Final[int]

            def __init__(self, n_iter: int):
                super().__init__()
                self.n_iter = n_iter

            def forward(self, img: torch.Tensor, psf: torch.Tensor) -> torch.Tensor:
                """Run the deconvolution."""
                psf_fft = torch.fft.rfft2(psf)  # [k, n, n/2+1]
                psft_fft = torch.fft.rfft2(torch.flip(psf, (-2, -1)))  # [k, n, n/2+1]
                data = torch.ones_like(psf) * 0.5  # [k, n, n]

                for _ in range(self.n_iter):
                    data = rl_step(data, img, psf_fft, psft_fft)

                return data

        jitted_fn = torch.jit.script(
            RL(num_steps),
            example_inputs=(
                torch.zeros(*img_size, dtype=torch.float32),
                torch.zeros(*psf_size, dtype=torch.float32),
            ),
        )

        torch.jit.save(jitted_fn, out_path)


class TorchIO(BaseIO):
    """PyTorch implementation of the IO backend."""

    @staticmethod
    def open(filename: str | Path) -> ArrayLike:
        """Open a file and return it as a numpy array."""

        img = Image.open(filename)
        frames = []
        for i in range(img.n_frames):
            img.seek(i)
            frames.append(torch.from_numpy(np.array(img).astype(np.float32)))

        return torch.stack(frames, dim=0)  # [n_frames, h, w]

    @staticmethod
    def save(
        filename: str | Path,
        data: ArrayLike,  # [n_frames, h, w]
        format=None,
    ) -> None:
        """Save a numpy array to file. The file format is determined by the filename suffix when present."""
        img = Image.fromarray(np.array(data[0].numpy()))
        img.save(filename, format=format, save_all=True, append_images=[Image.fromarray(np.array(d)) for d in data[1:]])
