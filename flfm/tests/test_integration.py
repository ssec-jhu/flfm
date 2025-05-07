import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import flfm.cli
import flfm.io
import flfm.pytorch_io
import flfm.pytorch_restoration
import flfm.restoration
import flfm.util


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


@pytest.mark.parametrize(
    ("backend_str", "expected_restoration", "expected_io"),
    [
        ("jax", flfm.restoration, flfm.io),
        ("torch", flfm.pytorch_restoration, flfm.pytorch_io),
    ],
)
def test_validate_backend(
    backend_str: str,
    expected_restoration: type,
    expected_io: type,
) -> None:
    """Test the backend validation function."""
    restoration, io = flfm.cli._validate_backend("jax")
    assert restoration is flfm.restoration
    assert io is flfm.io

    restoration, io = flfm.cli._validate_backend("torch")
    assert restoration is flfm.pytorch_restoration
    assert io is flfm.pytorch_io


def test_validate_backend_invalid() -> None:
    """Test the backend validation function with an invalid backend."""
    with pytest.raises(ValueError, match="not supported"):
        flfm.cli._validate_backend("invalid_backend")


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
    img = flfm.io.open(rand_img_stream)
    psf = flfm.io.open(sim_psf_stream)

    # reconstruct the image
    reconstructed = flfm.restoration.richardson_lucy(img, psf, num_iter=1)

    # save the reconstructed image to a byte stream
    out_stream = io.BytesIO()
    flfm.io.save(out_stream, reconstructed, format="TIFF")
    out_stream.seek(0)

    # read the reconstructed image back in from the byte stream
    out_img = flfm.io.open(out_stream)

    # make a circle mask
    mask = flfm.util.make_circle_mask(r)
    out = flfm.util.crop_and_apply_circle_mask(sim_psf, (h // 2, w // 2), r)

    expected_area = np.ceil(np.pi * r**2)
    assert img.shape == (1, h, w)
    assert psf.shape == (n, h, w)
    assert reconstructed.shape == (n, h, w)
    assert out_img.shape == (n, h, w)
    assert expected_area == mask.sum()
    assert mask.shape == (2 * r, 2 * r)
    assert out.shape == (n, 2 * r, 2 * r)


def test_simple_integration_pytorch() -> None:
    """Test the integration of the package."""
    r = 5
    n, h, w = 4, 32, 32

    # setup a random image and psf and save them to a byte stream
    rand_img = np.ones([1, h, w], dtype=np.float32)
    rand_img_stream = arr_to_stream(rand_img)

    sim_psf = np.ones([n, h, w], dtype=np.float32) / (w * h)
    sim_psf_stream = arr_to_stream(sim_psf)

    # read them back in from the byte stream
    img = flfm.pytorch_io.open(rand_img_stream)
    psf = flfm.pytorch_io.open(sim_psf_stream)

    # reconstruct the image
    reconstructed = flfm.pytorch_restoration.richardson_lucy(img, psf, num_iter=1)

    # save the reconstructed image to a byte stream
    out_stream = io.BytesIO()
    flfm.pytorch_io.save(out_stream, reconstructed, format="TIFF")
    out_stream.seek(0)

    # read the reconstructed image back in from the byte stream
    out_img = flfm.pytorch_io.open(out_stream)

    # make a circle mask
    mask = flfm.util.make_circle_mask(r)
    out = flfm.util.crop_and_apply_circle_mask(sim_psf, (h // 2, w // 2), r)

    expected_area = np.ceil(np.pi * r**2)
    assert img.shape == (1, h, w)
    assert psf.shape == (n, h, w)
    assert reconstructed.shape == (n, h, w)
    assert out_img.shape == (n, h, w)
    assert expected_area == mask.sum()
    assert mask.shape == (2 * r, 2 * r)
    assert out.shape == (n, 2 * r, 2 * r)


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_cli_main(tmp_path: Path, backend: str) -> None:
    """Test the command line interface."""
    r = 5
    n, h, w = 4, 32, 32

    # setup a random image and psf and save them to a byte stream
    rand_img = np.ones([1, h, w], dtype=np.float32)
    rand_img_stream = arr_to_stream(rand_img)

    sim_psf = np.ones([n, h, w], dtype=np.float32) / (w * h)
    sim_psf_stream = arr_to_stream(sim_psf)

    out_stream = tmp_path / "test_file.tiff"

    flfm.cli.main(
        img=rand_img_stream,
        psf=sim_psf_stream,
        out=out_stream,
        lens_radius=r,
        num_iters=1,
        backend=backend,
    )

    out_img = flfm.io.open(out_stream)

    assert out_img.shape == (n, 2 * r, 2 * r)


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_cli_export(tmp_path: Path, backend: str) -> None:
    """Test the export command line interface."""
    n_steps = 10
    out_stream = tmp_path / ("exported_model" + (".pt" * (backend == "torch")))

    flfm.cli.export(
        out=out_stream,
        n_steps=n_steps,
        backend=backend,
        img_size=(1, 2048, 2048),
        psf_size=(41, 2048, 2048),
    )

    assert out_stream.exists()
    if backend == "torch":
        assert out_stream.is_file() and out_stream.suffix == ".pt"
    else:
        assert not out_stream.is_file()
