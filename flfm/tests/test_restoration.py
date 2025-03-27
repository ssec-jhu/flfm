import flfm.io
import flfm.pytorch_io
import flfm.pytorch_restoration
import flfm.restoration
import flfm.util


class TestRichardsonLucy:
    def test_2d_image(self):
        image = flfm.io.open(flfm.util.find_repo_location() / "data/yale/light_field_image.tif")
        psf = flfm.io.open(flfm.util.find_repo_location() / "data/yale/measured_psf.tif")
        recon = flfm.restoration.richardson_lucy(image, psf[21])
        assert recon.shape == image.shape

    def test_2d_image_pytorch(self):
        image = flfm.pytorch_io.open(flfm.util.find_repo_location() / "data/yale/light_field_image.tif")
        psf = flfm.pytorch_io.open(flfm.util.find_repo_location() / "data/yale/measured_psf.tif")
        recon = flfm.pytorch_restoration.richardson_lucy(image, psf[21])
        assert recon.shape == image.shape
