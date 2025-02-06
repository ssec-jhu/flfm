import io

import numpy as np
from PIL import Image

import flfm.cli as flfm_cli
import flfm.io as flfm_io
import flfm.reconstruct as flfm_reconstruct
import flfm.util as flfm_utils


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


def test_simple_integration() -> None:
    """Test the integration of the package."""
    r = 5
    n, h, w = 4, 32, 32

    # setup a random image and psf and save them to a byte stream
    rand_img = np.ones([1, h, w], dtype=np.float32)
    rand_img_stream = arr_to_stream(rand_img)

    sim_psf = np.ones([n, h, w], dtype=np.float32) / (w * h)
    sim_psf_stream = arr_to_stream(sim_psf)

    # read them back in from the byte stream
    img = flfm_io.open_tiff(rand_img_stream)
    psf = flfm_io.open_tiff(sim_psf_stream)

    # reconstruct the image
    reconstructed = flfm_reconstruct.reconstruct(img, psf, num_iter=1)

    # save the reconstructed image to a byte stream
    out_stream = io.BytesIO()
    flfm_io.save_tiff(out_stream, reconstructed)
    out_stream.seek(0)

    # read the reconstructed image back in from the byte stream
    out_img = flfm_io.open_tiff(out_stream)

    # make a circle mask
    mask = flfm_utils.make_circle_mask(r)
    out = flfm_utils.crop_and_apply_circle_mask(sim_psf, (h // 2, w // 2), r)

    expected_area = np.ceil(np.pi * r**2)
    assert img.shape == (1, h, w)
    assert psf.shape == (n, h, w)
    assert reconstructed.shape == (n, h, w)
    assert out_img.shape == (n, h, w)
    assert expected_area == mask.sum()
    assert mask.shape == (2 * r, 2 * r)
    assert out.shape == (n, 2 * r, 2 * r)


def test_cli() -> None:
    """Test the command line interface."""
    r = 5
    n, h, w = 4, 32, 32

    # setup a random image and psf and save them to a byte stream
    rand_img = np.ones([1, h, w], dtype=np.float32)
    rand_img_stream = arr_to_stream(rand_img)

    sim_psf = np.ones([n, h, w], dtype=np.float32) / (w * h)
    sim_psf_stream = arr_to_stream(sim_psf)

    out_stream = io.BytesIO()

    flfm_cli.main(
        img=rand_img_stream,
        psf=sim_psf_stream,
        out=out_stream,
        lens_radius=r,
        num_iters=1,
    )

    out_img = flfm_io.open_tiff(out_stream)

    assert out_img.shape == (n, 2 * r, 2 * r)
