"""Base classes for the FLFM backend."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Sequence

from numpy.typing import ArrayLike

from flfm.settings import settings


class Singleton:
    """A singleton class."""

    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance


class BaseRestoration(Singleton, ABC):
    """Base class for the restoration backend."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.compiled_rl_step = None

    def compute_step(self, *args, **kwargs):
        """Wrapper for calling ``self.compiled_func()``. Override where needed."""
        return self.compiled_rl_step(*args, **kwargs)

    @staticmethod
    @abstractmethod
    def rfft2(a: ArrayLike, *args, axis: Sequence[int] = (-2, -1), **kwargs) -> ArrayLike:
        """Compute the 2D real-to-complex FFT."""
        ...

    @staticmethod
    @abstractmethod
    def ones_like(a: ArrayLike, *args, **kwargs) -> ArrayLike:
        """Return an array of ones with the same shape and type as a given array."""
        ...

    @staticmethod
    @abstractmethod
    def flip(a: ArrayLike, *args, axis: Sequence[int] = None, **kwargs) -> ArrayLike:
        """Reverse the order of elements in an array along the given axes."""
        ...

    @staticmethod
    @abstractmethod
    def sum(a: ArrayLike, *args, axis: Sequence[int] = (1, 2), keepdims: bool = True, **kwargs):
        """Sum of array elements over a given axis."""
        ...

    @staticmethod
    @abstractmethod
    def export_model(
        out_path: str | Path,
        num_steps: int,
        img_size: tuple[int, int, int],
        psf_size: tuple[int, int, int],
    ) -> None:
        """Export a model for use elsewhere."""
        ...

    @staticmethod
    @abstractmethod
    def to_device(data: ArrayLike, device: str | Any = None) -> ArrayLike:
        """Move data to a device."""
        ...

    def richardson_lucy_core(
        self,
        image: ArrayLike,  # [1, n, n]
        psf: ArrayLike,  # [k, n, n]
        num_iter: int = settings.DEFAULT_RL_ITERS,
    ) -> ArrayLike:
        """Core of the Richardson-Lucy deconvolution algorithm.

        Args:
            image: The image to deconvolve.
            psf: The point spread function.
            num_iter: The number of iterations to run.

        Returns:
            The deconvolved image.
        """
        psf_fft = self.rfft2(psf, axis=(-2, -1))  # [k, n, n/2+1]
        psft_fft = self.rfft2(self.flip(psf, axis=(-2, -1)))  # [k, n, n/2+1]
        data = self.ones_like(psf) * 0.5  # [k, n, n]

        for _ in range(num_iter):
            data = self.compute_step(data, image, psf_fft, psft_fft)

        return data

    def richardson_lucy(
        self,
        image: ArrayLike,  # [1, n, n]
        psf: ArrayLike,  # [k, n, n]
        num_iter: int = settings.DEFAULT_RL_ITERS,
        **kwargs,
    ) -> ArrayLike:
        """Reconstruct the image using the Richardson-Lucy deconvolution method.

        Args:
            image: The image to deconvolve.
            psf: The point spread function.
            num_iter: The number of iterations to run.
            **kwargs: Additional keyword arguments.

        Returns:
            The deconvolved image.
        """

        if "clip" in kwargs or "filter_epsilon" in kwargs:
            raise NotImplementedError

        # Copy data to device.
        image = self.to_device(image)
        psf = self.to_device(psf)

        # Do Richardson-Lucy deconvolution.
        data = self.richardson_lucy_core(image, psf, num_iter=num_iter)

        # Copy data from device.
        data = self.to_device(data, "cpu")  # TODO: Is there a use case for this in reconstruct rather than here?

        return data


class BaseIO(Singleton, ABC):
    """Base class for the IO backend."""

    @staticmethod
    @abstractmethod
    def open(filename: str | Path) -> ArrayLike:
        """Open a file and return it as a numpy array."""
        ...

    @staticmethod
    @abstractmethod
    def save(
        filename: str | Path,
        data: ArrayLike,  # [n_frames, h, w]
        format=None,
    ) -> None:
        """Save a numpy array to file. The file format is determined by the filename suffix when present."""
        ...
