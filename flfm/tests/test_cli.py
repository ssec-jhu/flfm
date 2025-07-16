from pathlib import Path

import numpy as np
import pytest

import flfm.backend.jax
import flfm.backend.torch
import flfm.cli
import flfm.util
from flfm.tests.conftest import arr_to_stream


@pytest.mark.parametrize("backend", ["torch", "jax"])
class TestCli:
    def test_cli_main(self, tmp_path: Path, backend: str) -> None:
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

    def test_cli_export(self, tmp_path: Path, backend: str) -> None:
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

    def test_cli_batch(self, tmp_path: Path, backend: str):
        """Test the batch command line interface."""
        out_dir = tmp_path / "batched_files"

        # Assert out dir starts empty.
        assert not list(out_dir.glob("*"))

        filename_pattern = "light*.tif"
        input_dir = flfm.util.find_package_location() / "tests" / "data" / "yale"
        flfm.cli.batch(
            input_dir, out_dir, input_dir / "measured_psf.tif", filename_pattern=filename_pattern, n_workers=1
        )

        assert len(list(out_dir.glob("*"))) == len(list(input_dir.glob(filename_pattern)))
