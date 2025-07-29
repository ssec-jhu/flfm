import io
import sys

import numpy as np
import pytest
from PIL import Image


def arr_to_stream(arr: np.ndarray) -> io.BytesIO:
    """Convert a numpy array to a bytes stream."""
    stream = io.BytesIO()

    if arr.ndim == 2:
        arr = np.expand_dims(arr, axis=0)

    Image.fromarray(arr[0]).save(
        stream,
        format="TIFF",
        save_all=True,
        append_images=[Image.fromarray(np.array(arr[i])) for i in range(1, len(arr))],
    )
    stream.seek(0)
    return stream


@pytest.fixture(autouse=True)
def reset_settings():
    from flfm.settings import settings

    settings.__init__()  # Reset. See https://docs.pydantic.dev/latest/concepts/pydantic_settings/#in-place-reloading
    for k, field in settings.__class__.model_fields.items():
        assert getattr(settings, k) == field.default


@pytest.fixture(autouse=True, scope="module")
def clear_all_imports():
    sys.modules.clear()
