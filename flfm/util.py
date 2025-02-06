import importlib
import os
from pathlib import Path

import jax.numpy as jnp

from . import __project__  # Keep as relative for templating reasons.


def find_package_location(package=__project__):
    return Path(importlib.util.find_spec(package).submodule_search_locations[0])


def find_repo_location(package=__project__):
    return Path(find_package_location(package) / os.pardir)


def make_circle_mask(radius: int) -> jnp.ndarray:
    """Create a circular mask."""
    y, x = jnp.ogrid[: 2 * radius, : 2 * radius]
    circle_mask = (x - radius) ** 2 + (y - radius) ** 2 <= radius**2
    return circle_mask.astype(jnp.float32)  # [2 * radius, 2 * radius]


def crop_and_apply_circle_mask(
    data: jnp.ndarray,  # [k, n, n]
    center: tuple[int, int],
    radius: int,
) -> jnp.ndarray:
    """Crop the image and apply a circular mask."""
    circle_mask = jnp.expand_dims(make_circle_mask(radius), axis=0)  # [1, 2 * radius, 2 * radius]
    sub_O = data[
        :, center[0] - radius : center[0] + radius, center[1] - radius : center[1] + radius
    ]  # [k, 2 * radius, 2 * radius]
    return sub_O * circle_mask  # [k, 2 * radius, 2 * radius]
