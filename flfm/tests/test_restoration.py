import flfm.io
import flfm.pytorch_io
import flfm.pytorch_restoration
import flfm.restoration
import flfm.util

data_dir = flfm.util.find_package_location() / "tests" / "data" / "yale"
input_filename = data_dir / "light_field_image.tif"
psf_filename = data_dir / "measured_psf.tif"


class TestRichardsonLucy:
    def test_2d_image(self):
        image = flfm.io.open(input_filename)
        psf = flfm.io.open(psf_filename)
        recon = flfm.restoration.richardson_lucy(image, psf[21])
        assert recon.shape == image.shape

    def test_2d_image_pytorch(self):
        image = flfm.pytorch_io.open(input_filename)
        psf = flfm.pytorch_io.open(psf_filename)
        recon = flfm.pytorch_restoration.richardson_lucy(image, psf[21])
        assert recon.shape == image.shape
