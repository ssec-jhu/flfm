import flfm.io
import flfm.restoration
import flfm.util


class TestRichardsonLucy:
    def test_2d_image(self):
        image = flfm.io.open(flfm.util.find_repo_location() / "data/yale/light_field_image.tif")
        psf = flfm.io.open(flfm.util.find_repo_location() / "data/yale/measured_psf.tif")
        recon = flfm.restoration.reconstruct(image, psf[21])
        assert recon.shape == image.shape
