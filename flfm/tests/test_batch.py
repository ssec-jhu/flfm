import multiprocessing as mp
import os
import shutil
from pathlib import Path

import pytest
from dask.distributed import LocalCluster

import flfm.io
import flfm.util
from flfm.batch import batch_reconstruction

data_dir = flfm.util.find_repo_location() / "data" / "yale"
input_filename = data_dir / "light_field_image.tif"
psf_filename = data_dir / "measured_psf.tif"
n_copies = 5
n_workers = 1 if os.environ.get("CI") else 2


def mock_batch_data(image_filename: str | Path, output_dir: str | Path, n_copies: int):
    """This helper func can be used to create mock data sets/dirs for, e.g., perf testing."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_filename = Path(image_filename)

    def do_copy(input_filename, output_filename):
        shutil.copy(input_filename, output_filename)

    with LocalCluster(n_workers=mp.cpu_count(), threads_per_worker=n_workers) as cluster:
        client = cluster.get_client()

        futures = []
        for i in range(n_copies):
            future = client.submit(
                do_copy,
                image_filename,
                output_dir / f"{image_filename.stem}_{i + 1}{image_filename.suffix}",
            )
            futures.append(future)

        client.gather(futures)


@pytest.fixture
def mock_data(tmp_path):
    output_dir = tmp_path / "input"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Assert empty.
    assert not flfm.util.find_files(output_dir)

    mock_batch_data(input_filename, output_dir, n_copies)
    return output_dir


@pytest.mark.skipif(os.environ.get("CI") == "true", reason="Runs out of memory.")
class TestBatchReconstruction:
    def test_mock_data(self, mock_data):
        assert len(flfm.util.find_files(mock_data)) == n_copies

    def test_batch_reconstruction(self, mock_data):
        input_dir = mock_data
        output_dir = mock_data.parent / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Assert empty.
        assert not flfm.util.find_files(output_dir)

        # Reduce memory consumption by only using half the # of CPUs.
        processed_files = batch_reconstruction(input_dir, output_dir, psf_filename, clobber=True, n_workers=n_workers)
        assert len(flfm.util.find_files(mock_data)) == n_copies
        assert len(processed_files) == n_copies

    def test_carry_on(self, mock_data):
        input_dir = mock_data
        output_dir = mock_data.parent / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Assert empty.
        assert not flfm.util.find_files(output_dir)

        # Copy half data to mock in complete initial run.
        files = flfm.util.find_files(input_dir)
        for i, filename in enumerate(files):
            if i >= len(files) // 2:
                break
            shutil.copy(filename, output_dir / filename.name)

        assert len(flfm.util.find_files(output_dir)) == n_copies // 2

        # Reduce memory consumption by only using half the # of CPUs.
        processed_files = batch_reconstruction(
            input_dir, output_dir, psf_filename, clobber=True, n_workers=2, carry_on=True
        )

        assert len(flfm.util.find_files(mock_data)) == n_copies
        assert len(processed_files) == len(files) - len(files) // 2
