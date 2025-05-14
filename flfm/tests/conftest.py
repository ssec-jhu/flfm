import io

import numpy as np
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
