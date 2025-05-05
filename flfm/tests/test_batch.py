import multiprocessing as mp
import shutil

import pytest

import flfm.io
import flfm.util
from flfm.batch import batch_reconstruction, mock_batch_data
from flfm.settings import settings

# TODO: Refactor this to some __init__.py. See https://github.com/ssec-jhu/flfm/issues/146.
match settings.BACKEND:
    case "torch":
        import flfm.pytorch_io as flfm_io
    case "jax":
        import flfm.io as flfm_io
    case _:
        raise NotImplementedError(f"Backend {settings.BACKEND} not implemented.")


data_dir = flfm.util.find_repo_location() / "data" / "yale"
input_filename = data_dir / "light_field_image.tif"
n_copies = 5


@pytest.fixture
def mock_data(tmp_path):
    output_dir = tmp_path / "input"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Assert empty.
    assert not flfm.util.find_files(output_dir)

    mock_batch_data(input_filename, output_dir, n_copies)
    return output_dir


class TestBatch:
    def test_mock_data(self, mock_data):
        assert len(flfm.util.find_files(mock_data)) == n_copies

    def test_batch_reconstruction(self, mock_data):
        psf = flfm_io.open(data_dir / "measured_psf.tif")
        input_dir = mock_data
        output_dir = mock_data.parent / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Assert empty.
        assert not flfm.util.find_files(output_dir)

        # Reduce memory consumption by only using half the # of CPUs.
        processed_files = batch_reconstruction(input_dir, output_dir, psf, clobber=True, n_workers=mp.cpu_count() // 2)
        assert len(flfm.util.find_files(mock_data)) == n_copies
        assert len(processed_files) == n_copies

    def test_carry_on(self, mock_data):
        psf = flfm_io.open(data_dir / "measured_psf.tif")
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
            input_dir, output_dir, psf, clobber=True, n_workers=mp.cpu_count() // 2, carry_on=True
        )

        assert len(flfm.util.find_files(mock_data)) == n_copies
        assert len(processed_files) == len(files) - len(files) // 2
