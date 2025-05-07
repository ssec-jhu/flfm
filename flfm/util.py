import importlib
import os
from pathlib import Path

import numpy as np

from . import __project__  # Keep as relative for templating reasons.


def find_package_location(package=__project__):
    return Path(importlib.util.find_spec(package).submodule_search_locations[0])


def find_repo_location(package=__project__):
    return Path(find_package_location(package) / os.pardir)


def make_circle_mask(radius: int) -> np.ndarray:
    """Create a circular mask."""
    y, x = np.ogrid[: 2 * radius, : 2 * radius]
    circle_mask = (x - radius) ** 2 + (y - radius) ** 2 <= radius**2
    return circle_mask.astype(np.float32)  # [2 * radius, 2 * radius]


def crop_and_apply_circle_mask(
    data: np.ndarray,  # [k, n, n]
    center: tuple[int, int],
    radius: int,
) -> np.ndarray:
    """Crop the image and apply a circular mask."""
    center = [int(x) for x in center]
    radius = int(radius)
    circle_mask = np.expand_dims(make_circle_mask(radius), axis=0)  # [1, 2 * radius, 2 * radius]
    sub_O = data[
        :, center[0] - radius : center[0] + radius, center[1] - radius : center[1] + radius
    ]  # [k, 2 * radius, 2 * radius]
    return sub_O * circle_mask  # [k, 2 * radius, 2 * radius]


def find_files(directory: str | Path, ext=None):
    """Find all files in a directory, filter on ext if given."""
    directory = Path(directory)
    file_list = []
    for file in directory.iterdir():
        if (ext is not None) and (not file.suffix == ext):
            continue
        full_path = directory / file
        if full_path.is_file():
            file_list.append(full_path.absolute())
    return file_list
