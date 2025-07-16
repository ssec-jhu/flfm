"""Utility functions for FLFM."""

import importlib
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np

from flfm import __project__
from flfm.settings import settings


def find_package_location(package: str = __project__) -> Path:
    """Find the location of an installed package.

    Args:
        package: The name of the package.

    Returns:
        The path to the package.
    """
    return Path(importlib.util.find_spec(package).submodule_search_locations[0])


def find_repo_location(package: str = __project__) -> Path:
    """Find the location of the repository.

    Args:
        package: The name of the package.

    Returns:
        The path to the repository.
    """
    return Path(find_package_location(package) / os.pardir)


def make_circle_mask(radius: int) -> np.ndarray:
    """Create a circular mask.

    Args:
        radius: The radius of the circle.

    Returns:
        A circular mask.
    """
    y, x = np.ogrid[: 2 * radius, : 2 * radius]
    circle_mask = (x - radius) ** 2 + (y - radius) ** 2 <= radius**2
    return circle_mask.astype(np.float32)  # [2 * radius, 2 * radius]


def crop_and_apply_circle_mask(
    data: np.ndarray,  # [k, n, n]
    center: tuple[int, int],
    radius: int,
) -> np.ndarray:
    """Crop the image and apply a circular mask.

    Args:
        data: The image data.
        center: The center of the circle.
        radius: The radius of the circle.

    Returns:
        The cropped and masked image.
    """
    center = [int(x) for x in center]
    radius = int(radius)
    circle_mask = np.expand_dims(make_circle_mask(radius), axis=0)  # [1, 2 * radius, 2 * radius]
    sub_O = data[
        :, center[0] - radius : center[0] + radius, center[1] - radius : center[1] + radius
    ]  # [k, 2 * radius, 2 * radius]
    return sub_O * circle_mask  # [k, 2 * radius, 2 * radius]


def find_files(directory: str | Path, ext: str | None = None) -> list[Path]:
    """Find all files in a directory, filter on ext if given.

    Args:
        directory: The directory to search.
        ext: The extension to filter on.

    Returns:
        A list of files.
    """
    directory = Path(directory)
    file_list = []
    for file in directory.iterdir():
        if (ext is not None) and (not file.suffix == ext):
            continue
        full_path = directory / file
        if full_path.is_file():
            file_list.append(full_path.absolute())
    return file_list


def setup_logging(level: str = settings.LOG_LEVEL, format: str = settings.LOG_FORMAT, **kwargs):
    """Set up logging.

    Args:
        level: The logging level.
        format: The logging format.
        **kwargs: Additional keyword arguments to pass to `logging.basicConfig`.
    """
    logging.basicConfig(level=level, format=format, **kwargs)


def get_latest_filename(directory: str | Path, ext: str = ".tif") -> Optional[Path]:
    """Get the latest filename from a directory.

    Args:
        directory: The directory to search.
        ext: The extension to filter on.

    Returns:
        The latest filename, or None if no files are found.
    """
    directory = Path(directory)
    files = list(directory.glob(f"*{ext}"))
    if not files:
        return None
    latest_file = max(files, key=lambda x: x.stat().st_mtime)
    return latest_file
