import pytest

import flfm.util

DATA_DIR = flfm.util.find_package_location() / "tests" / "data" / "yale"
PSF_FILENAME = DATA_DIR / "measured_psf.tif"
IMAGE_FILENAME = DATA_DIR / "light_field_image.tif"


@pytest.mark.parametrize("filename", (PSF_FILENAME, IMAGE_FILENAME))
def test_data_access(filename):
    assert filename.exists()
