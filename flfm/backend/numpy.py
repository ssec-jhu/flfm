from pathlib import Path
from typing import Any, Sequence

import numpy as np
from numpy.typing import ArrayLike
from PIL import Image

from flfm.backend.base import BaseIO, BaseRestoration


def rl_step(
    data: np.ndarray,  # [k, n, n]
    image: np.ndarray,  # [1, n, n]
    PSF_fft: np.ndarray,  # [k, n, n/2+1]
    PSFt_fft: np.ndarray,  # [k, n, n/2+1]
) -> np.ndarray:
    """Single step of the multiplicative Richardson-Lucy deconvolution algorithm.

    Args:
        data: The current estimate of the deconvolved image.
        image: The observed image.
        PSF_fft: The FFT of the point spread function.
        PSFt_fft: The FFT of the time-reversed point spread function.

    Returns:
        The next estimate of the deconvolved image.
    """
    # NOTE: This could use ``cls.rfft2``, however, we don't so that we can keep this func static.
    denominator = np.fft.irfft2(PSF_fft * np.fft.rfft2(data)).sum(axis=0, keepdims=True)  # [1, n, n]
    img_err = image / denominator
    return data * np.fft.fftshift(
        np.fft.irfft2(np.fft.rfft2(img_err) * PSFt_fft), axes=(-2, -1)
    )  # [k, n, n]


class NumpyRestoration(BaseRestoration):
    """Numpy implementation of the restoration backend."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.compiled_rl_step = rl_step

    def compute_step(self, *args, **kwargs):
        """Wrapper for calling ``self.compiled_func()``. Override where needed."""
        return self.compiled_rl_step(*args, **kwargs)

    @staticmethod
    def rfft2(a: ArrayLike, *args, axis: Sequence[int] = (-2, -1), **kwargs):
        """Compute the 2D real-to-complex FFT."""
        return np.fft.rfft2(a, *args, axes=axis, **kwargs)

    @staticmethod
    def ones_like(a: ArrayLike, *args, **kwargs) -> ArrayLike:
        """Return an array of ones with the same shape and type as a given array."""
        return np.ones_like(a, *args, **kwargs)

    @staticmethod
    def flip(a: ArrayLike, *args, axis: Sequence[int] = None, **kwargs) -> ArrayLike:
        """Reverse the order of elements in an array along the given axes."""
        return np.flip(a, *args, axis=axis, **kwargs)

    @staticmethod
    def sum(a: ArrayLike, *args, axis: Sequence[int] = (1, 2), keepdims: bool = True, **kwargs):
        """Sum of array elements over a given axis."""
        return np.sum(a, axis=axis, keepdims=keepdims)

    @staticmethod
    def to_device(data: ArrayLike, device: str | Any = None) -> ArrayLike:
        """Move data to a device."""
        return data

    @staticmethod
    def export_model(
            out_path: str | Path,
            num_steps: int,
            img_size: tuple[int, int, int],
            psf_size: tuple[int, int, int],
    ) -> None:
        """Export a model for use elsewhere."""
        raise NotImplementedError


class NumpyIO(BaseIO):
    """Numpy implementation of the IO backend."""

    @staticmethod
    def open(filename: str | Path) -> ArrayLike:
        """Open a file and return it as a numpy array."""
        img = Image.open(filename)
        frames = []
        for i in range(img.n_frames):
            img.seek(i)
            frames.append(np.array(img))

        return np.stack(frames, axis=0)  # [n_frames, h, w]

    @staticmethod
    def save(
        filename: str | Path,
        data: ArrayLike,  # [n_frames, h, w]
        format=None,
    ) -> None:
        """Save a numpy array to file. The file format is determined by the filename suffix when present."""
        img = Image.fromarray(np.array(data[0]))
        img.save(filename, format=format, save_all=True, append_images=[Image.fromarray(np.array(d)) for d in data[1:]])
