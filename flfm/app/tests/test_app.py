import base64
import time
from copy import deepcopy
from multiprocessing import Process

import numpy as np
import pytest
import requests
from dash import dcc, html
from plotly.graph_objs import Figure

import flfm.app.main
import flfm.io
import flfm.util
from flfm.settings import app_settings, settings

DATA_DIR = flfm.util.find_repo_location() / "data" / "yale"
PSF_FILENAME = DATA_DIR / "measured_psf.tif"
IMAGE_FILENAME = DATA_DIR / "light_field_image.tif"
IDS = ("psf", "image", "reconstruction")
FILENAMES = {IDS[0]: PSF_FILENAME, IDS[1]: IMAGE_FILENAME}


def start_real_server():
    # Start a real server on a separate process. It's easier to kill a process than a thread.
    # For the majority of tests we can just use ``dash.testing`` or ``fastapi.testclient`` instead of this. However, it
    # would be good to still test prd deployment run code.
    proc = Process(target=flfm.app.main.start_app)
    proc.start()
    time.sleep(5)  # Both the new process & server take time to start up.

    if not proc.is_alive():
        raise RuntimeError("Server did not start")

    def _kill():
        proc.kill()
        proc.join()

    return _kill


class TestRunServer:
    def test_wrong_api(self, monkeypatch):
        monkeypatch.setattr(app_settings, "WEB_API", "blah")
        with pytest.raises(NotImplementedError):
            flfm.app.main.start_app()

    def test_flask_debug_mode_exception(self, monkeypatch):
        monkeypatch.setattr(app_settings, "WEB_API", "flask")
        monkeypatch.setattr(app_settings, "DEBUG", False)

        with pytest.raises(NotImplementedError):
            flfm.app.main.start_app()

    @pytest.mark.parametrize(("api"), ("flask", "fastapi"))
    def test_run_server(self, api, monkeypatch):
        monkeypatch.setattr(app_settings, "WEB_API", api)

        if app_settings.WEB_API == "flask":
            monkeypatch.setattr(app_settings, "DEBUG", api)

        url = f"http://{app_settings.HOST}:{app_settings.PORT}"

        # Assert server is not already running.
        with pytest.raises(requests.exceptions.ConnectionError):
            requests.get(url)

        stop_server = None
        try:
            stop_server = start_real_server()

            response = requests.get(url)
            assert response.status_code == 200
        finally:
            if stop_server:
                stop_server()


@pytest.fixture
def image_data():
    flfm.app.main.image_data["psf"] = flfm.app.main.flfm_io.open(PSF_FILENAME)
    flfm.app.main.image_data["image"] = flfm.app.main.flfm_io.open(IMAGE_FILENAME)
    flfm.app.main.image_data["uncropped_reconstruction"] = flfm.app.main.flfm_io.open(IMAGE_FILENAME)  # Spoof.
    flfm.app.main.image_data["reconstruction"] = flfm.app.main.flfm_io.open(IMAGE_FILENAME)  # Spoof.

    yield

    for k in flfm.app.main.image_data:
        flfm.app.main.image_data[k] = None


@pytest.fixture
def tooltips():
    slider_values = [10, 15]
    center_frame = 20
    tooltips = [
        {"template": flfm.app.main.slider_template_str(x, center_frame, app_settings.DEPTH_STEP_SIZE)}
        for x in slider_values
    ]
    return tooltips, slider_values, center_frame


class TestCallbacks:
    @pytest.mark.parametrize("index", IDS)
    def test_update_color_scale(self, index, image_data):
        fig = flfm.app.main.update_color_scale("plasma", 0, dict(index=index))
        assert isinstance(fig, Figure)

    def test_update_crop(self, image_data):
        cropped_reconstruction = flfm.app.main.image_data["reconstruction"]
        assert flfm.app.main.image_data["reconstruction"] is cropped_reconstruction
        uncropped_reconstruction = flfm.app.main.image_data["uncropped_reconstruction"]
        assert flfm.app.main.image_data["uncropped_reconstruction"] is uncropped_reconstruction

        fig = flfm.app.main.update_crop(
            settings.DEFAULT_CENTER_X + 10, settings.DEFAULT_CENTER_Y + 10, settings.DEFAULT_RADIUS + 10, 0
        )
        assert isinstance(fig, Figure)
        assert flfm.app.main.image_data["reconstruction"] is not cropped_reconstruction
        assert flfm.app.main.image_data["uncropped_reconstruction"] is uncropped_reconstruction

    def test_normalize_psf(self, image_data):
        psf_original = deepcopy(flfm.app.main.image_data["psf"])
        fig, disabled = flfm.app.main.normalize_psf(1, 20, app_settings.IMSHOW_COLOR_SCALE)
        assert isinstance(fig, Figure)
        assert disabled
        assert not np.array_equal(flfm.app.main.image_data["psf"], psf_original)

    @staticmethod
    def assert_graph_children(children: list, id: str):
        assert isinstance(children[0], dcc.Graph)
        assert children[0].id == dict(type="image-graph", index=id)
        assert isinstance(children[1], html.Label)
        assert isinstance(children[2], dcc.Slider)
        assert children[2].id == dict(type="image-slider", index=id)
        assert isinstance(children[3], html.Label)
        assert isinstance(children[4], dcc.Dropdown)
        assert children[4].id == dict(type="color-scale", index=id)
        assert children[4].options == flfm.app.main.colorscales

    def test_reconstruct(self, image_data):
        id = "reconstruction"
        flfm.app.main.image_data[id] = None
        children = flfm.app.main.reconstruct(
            1, 2, settings.DEFAULT_CENTER_X, settings.DEFAULT_CENTER_Y, settings.DEFAULT_RADIUS
        )
        assert flfm.app.main.image_data[id] is not None
        self.assert_graph_children(children, id)

    def test_update_slider_tooltip(self, tooltips):
        tooltips, slider_values, center_frame = tooltips
        new_tooltips = flfm.app.main.update_slider_tooltip(
            app_settings.DEPTH_STEP_SIZE + 10, slider_values, tooltips, 40
        )
        assert len(new_tooltips) == len(tooltips)
        assert new_tooltips == [
            {
                "template": flfm.app.main.slider_template_str(
                    x, center_frame, app_settings.DEPTH_STEP_SIZE + 10, unit=app_settings.DEPTH_UNIT
                )
            }
            for x in slider_values
        ]

    @pytest.mark.parametrize(("index", "expected"), [(x, {"template": "-50.00 µm"}) for x in IDS])
    def test_update_image_from_slider(self, image_data, index, expected, tooltips):
        tooltips, _slider_values, center_frame = tooltips
        new_fig, new_tooltip = flfm.app.main.update_image_from_slider(
            0, tooltips[0], center_frame, 5, app_settings.IMSHOW_COLOR_SCALE, dict(index=index)
        )
        assert isinstance(new_fig, Figure)
        assert new_tooltip == expected

    @staticmethod
    def file_contents(filename: str) -> str:
        """Spoof file contents as expected from browser."""
        with open(filename, "rb") as f:
            data = f.read()
        encoded_data = base64.b64encode(data)
        contents = f"content_type,{encoded_data.decode()}"
        return contents

    @pytest.mark.parametrize("index", IDS[:2])
    def test_upload_data(self, index):
        assert flfm.app.main.image_data[index] is None
        filename = FILENAMES[index]
        contents = self.file_contents(filename)

        children, n_frames_from_image = flfm.app.main.upload_data(contents, filename, dict(index=index))
        assert flfm.app.main.image_data[index] is not None
        self.assert_graph_children(children, index)
