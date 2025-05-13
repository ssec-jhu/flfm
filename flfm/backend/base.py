"""FLFM observation reconstruction."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from numpy.typing import ArrayLike

from flfm.settings import settings


class Singleton:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance


class BaseRestoration(Singleton, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.compiled_rl_step = None

    def compute_step(self, *args, **kwargs):
        """Wrapper for calling ``self.compiled_func()``. Override where needed."""
        return self.compiled_rl_step(*args, **kwargs)

    @staticmethod
    @abstractmethod
    def rfft2(a: ArrayLike, *args, axis: Any = (-2, -1), **kargs) -> ArrayLike: ...

    @staticmethod
    @abstractmethod
    def ones_like(a: ArrayLike, *args, **kwargs) -> ArrayLike: ...

    @staticmethod
    @abstractmethod
    def flip(a: ArrayLike, *args, axis: Any = None, **kwargs) -> ArrayLike: ...

    @staticmethod
    @abstractmethod
    def sum(a: ArrayLike, *args, axis: Any = (1, 2), keepdims: bool = True, **kwargs): ...

    @staticmethod
    @abstractmethod
    def export_tf_model(
        out_path: str | Path,
        num_steps: int,
        img_size: tuple[int, int, int],
        psf_size: tuple[int, int, int],
    ) -> None: ...

    @staticmethod
    def to_device(data, device: str | Any = None):
        return data

    def richardson_lucy(
        self,
        image: ArrayLike,  # [1, n, n]
        psf: ArrayLike,  # [k, n, n]
        num_iter: int = settings.DEFAULT_RL_ITERS,
        **kwargs,
    ) -> ArrayLike:
        """Reconstruct the image using the Richardson-Lucy deconvolution method."""

        if "clip" in kwargs or "filter_epsilon" in kwargs:
            raise NotImplementedError

        image = self.to_device(image)
        psf = self.to_device(psf)

        # We may want to make this something the use changes
        data = self.ones_like(psf) * 0.5  # [k, n, n]

        psf_fft = self.rfft2(psf, axis=(-2, -1))  # [k, n, n/2+1]
        psf_miror = self.flip(psf, axis=(-2, -1))
        psft_fft = self.rfft2(psf_miror)  # [k, n, n/2+1]

        for _ in range(num_iter):
            data = self.compute_step(data, image, psf_fft, psft_fft)

        data = self.to_device(data, "cpu")  # TODO: Is there a use case for this in reconstruct rather than here?

        return data


class BaseIO(Singleton, ABC):
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
