import base64
import io
import logging
from functools import partial

import dash_bootstrap_components as dbc
import plotly.express as px
from dash import ALL, MATCH, Dash, Input, Output, State, callback, dcc, html, no_update
from fastapi import FastAPI
from numpy.typing import ArrayLike

import flfm.util
from flfm.settings import app_settings, settings

# TODO: Refactor this to some __init__.py. See https://github.com/ssec-jhu/flfm/issues/146.
match settings.BACKEND:
    case "torch":
        import torch

        psf_sum = partial(torch.sum, dim=(1, 2), keepdim=True)
        import flfm.pytorch_io as flfm_io
        import flfm.pytorch_restoration as flfm_restoration
    case "jax":
        import jax.numpy as jnp

        psf_sum = partial(jnp.sum, axis=(1, 2), keepdims=True)
        import flfm.io as flfm_io
        import flfm.restoration as flfm_restoration
    case _:
        raise NotImplementedError(f"Backend {settings.BACKEND} not implemented.")


flfm.util.setup_logging(level=settings.LOG_LEVEL, format=settings.LOG_FORMAT)
logger = logging.getLogger(__name__)


def exception_handler(error):
    if app_settings.DEBUG:
        raise error
    else:
        logger.exception(error)


colorscales = px.colors.named_colorscales()


external_stylesheets = [getattr(dbc.themes, app_settings.THEME)]
dash_app = Dash(__name__, external_stylesheets=external_stylesheets, on_error=exception_handler)
dash_server = dash_app.server


def plot_image(data: ArrayLike, *args, color_scale: str = app_settings.IMSHOW_COLOR_SCALE, **kwargs):
    fig = px.imshow(data, *args, color_continuous_scale=color_scale, **kwargs)
    fig.update_xaxes(showticklabels=app_settings.IMSHOW_SHOW_TICK_LABELS)
    fig.update_yaxes(showticklabels=app_settings.IMSHOW_SHOW_TICK_LABELS)
    return fig


def run_dummy_reconstruction(*args, **kwargs):
    """Run a dummy reconstruction to get any jax/pytorch compilation overhead out of the way."""
    logger.info("Running dummy reconstruction...")
    psf = flfm_io.open(app_settings.DUMMY_PSF_FILEPATH)
    light_field_image = flfm_io.open(app_settings.DUMMY_LIGHT_FILED_IMAGE_FILEPATH)
    reconstruction = flfm_restoration.richardson_lucy(light_field_image, psf, *args, **kwargs)
    logger.info("Running dummy reconstruction... COMPLETED.")
    return reconstruction


def startup_tasks():
    if app_settings.RUN_DUMMY_RECONSTRUCTION_AT_STARTUP:
        run_dummy_reconstruction(num_iter=app_settings.STARTUP_DUMMY_RECONSTRUCTION_N_ITERS)


match app_settings.WEB_API:
    case "fastapi":
        from contextlib import asynccontextmanager

        from a2wsgi import WSGIMiddleware

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # Startup:
            startup_tasks()
            yield
            # Shutdown
            ...

        app = FastAPI(lifespan=lifespan)
        app.mount("/", WSGIMiddleware(dash_app.server))
    case "flask":
        ...
    case _:
        raise NotImplementedError(f"Unknown web API: {app_settings.WEB_API}")


# This is icky and will obviously break with multiple server instances. However, this is intended for local use, and
# from a single user. Alternative would be to use dcc.store, however, that only stores json and the image data would
# have to be loaded to the GPU multiple times unnecessarily.
# TODO: address? See https://github.com/ssec-jhu/flfm/issues/147.
image_data = dict(psf=None, image=None, reconstruction=None, uncropped_reconstruction=None)


def _center_frame(n_frames: int) -> int:
    return n_frames // 2


def slider_template_str(value: int, center: int, step_size: float, unit: str = app_settings.DEPTH_UNIT) -> str:
    return f"{(value - center) * step_size: .2f} {unit}"


def dash_layout():
    layout = html.Div(
        children=[
            dcc.Store(id=dict(type="n-frames", index="psf"), storage_type="memory", clear_data=True, data=0),
            # NOTE: the store below isn't really needed as "psf" is sufficient, however, MATCH has to match all outputs
            # in a callback so this is added so that ``upload_data()`` doesn't complain.
            dcc.Store(id=dict(type="n-frames", index="image"), storage_type="memory", clear_data=True, data=0),
            html.H1(children="FLFM 3D Reconstruction", style={"textAlign": "center"}),
            html.Br(),
            f"Depth step size ({app_settings.DEPTH_UNIT})",
            dbc.Input(
                value=app_settings.DEPTH_STEP_SIZE, type="number", id="depth-step-size", debounce=app_settings.DEBOUNCE
            ),
            html.Br(),
            html.Br(),
            dbc.Tabs(
                [
                    dbc.Tab(
                        label="PSF",
                        tabClassName="flex-grow-1 text-center",
                        children=[
                            html.Br(),
                            dcc.Upload(
                                id=dict(type="upload", index="psf"),
                                children=html.Div(["Drag and Drop PSF or ", html.A("Select Files")]),
                                style={
                                    "width": "100%",
                                    "height": "60px",
                                    "lineHeight": "60px",
                                    "borderWidth": "1px",
                                    "borderStyle": "dashed",
                                    "borderRadius": "5px",
                                    "textAlign": "center",
                                },
                                multiple=False,  # Allow only single file upload
                            ),
                            dcc.Loading(
                                id=dict(type="loading", index="psf"),
                                type=app_settings.LOADING_TYPE,
                                delay_show=app_settings.LOADING_DELAY_SHOW,
                                children=html.Div(id=dict(type="display", index="psf")),
                            ),
                            html.Br(),
                            html.Div(
                                [
                                    dbc.Button("Normalize PSF", id="normalize-psf", n_clicks=0),
                                ],
                                className="d-grid gap-2",
                            ),
                            html.Br(),
                        ],
                    ),
                    dbc.Tab(
                        label="Light-Field Image",
                        tabClassName="flex-grow-1 text-center",
                        children=[
                            html.Br(),
                            dcc.Upload(
                                id=dict(type="upload", index="image"),
                                children=html.Div(["Drag and Drop light-field-image or ", html.A("Select File")]),
                                style={
                                    "width": "100%",
                                    "height": "60px",
                                    "lineHeight": "60px",
                                    "borderWidth": "1px",
                                    "borderStyle": "dashed",
                                    "borderRadius": "5px",
                                    "textAlign": "center",
                                },
                                multiple=False,  # Allow only single file upload
                            ),
                            dcc.Loading(
                                id=dict(type="loading", index="image"),
                                type=app_settings.LOADING_TYPE,
                                delay_show=app_settings.LOADING_DELAY_SHOW,
                                children=html.Div(id=dict(type="display", index="image")),
                            ),
                        ],
                    ),
                    dbc.Tab(
                        label="3D Reconstruction",
                        tabClassName="flex-grow-1 text-center",
                        children=[
                            html.Br(),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        html.Div(
                                            [
                                                "Number of Iterations",
                                                dbc.Input(
                                                    value=settings.DEFAULT_RL_ITERS,
                                                    type="number",
                                                    id="rl-iters",
                                                    debounce=app_settings.DEBOUNCE,
                                                ),
                                            ]
                                        )
                                    ),
                                    dbc.Col(
                                        html.Div(
                                            [
                                                "Center x",
                                                dbc.Input(
                                                    value=settings.DEFAULT_CENTER_X,
                                                    type="number",
                                                    id="center_x",
                                                    debounce=app_settings.DEBOUNCE,
                                                ),
                                                "Center y",
                                                dbc.Input(
                                                    value=settings.DEFAULT_CENTER_Y,
                                                    type="number",
                                                    id="center_y",
                                                    debounce=app_settings.DEBOUNCE,
                                                ),
                                                "Radius",
                                                dbc.Input(
                                                    value=settings.DEFAULT_RADIUS,
                                                    type="number",
                                                    id="radius",
                                                    debounce=app_settings.DEBOUNCE,
                                                ),
                                            ]
                                        )
                                    ),
                                    dbc.Col(
                                        html.Div(
                                            [
                                                dbc.Button("Reconstruct", id="run-reconstruction", n_clicks=0),
                                            ]
                                        )
                                    ),
                                ]
                            ),
                            dcc.Loading(
                                id=dict(type="loading", index="reconstruction"),
                                type=app_settings.LOADING_TYPE,
                                delay_show=app_settings.LOADING_DELAY_SHOW,
                                children=html.Div(id=dict(type="display", index="reconstruction")),
                            ),
                        ],
                    ),
                ]
            ),
        ]
    )
    return layout


dash_app.layout = dash_layout


@callback(
    Output(dict(type="display", index=MATCH), "children", allow_duplicate=True),
    Output(dict(type="n-frames", index=MATCH), "data"),
    Input(dict(type="upload", index=MATCH), "contents"),
    State(dict(type="upload", index=MATCH), "filename"),
    State(dict(type="upload", index=MATCH), "id"),
    prevent_initial_call=True,
)
def upload_data(contents: str, filename: str, id: dict[str, str]):
    """Upload data/image files."""
    global image_data
    id = id["index"]
    image_data[id] = upload_file(contents, filename)
    children, n_frames_from_image = _display_image_with_slider(image_data[id], id=id)
    return children, n_frames_from_image


@callback(
    Output("normalize-psf", "disabled", allow_duplicate=True),
    Input(dict(type="upload", index="psf"), "last_modified"),
    prevent_initial_call=True,
)
def enable_normalization_button(_):
    """Re-enable normalization button upon new upload."""
    return False


@callback(
    Output(dict(type="image-graph", index=MATCH), "figure"),
    Output(dict(type="image-slider", index=MATCH), "tooltip"),
    Input(dict(type="image-slider", index=MATCH), "value"),
    State(dict(type="image-slider", index=MATCH), "tooltip"),
    State(dict(type="n-frames", index="psf"), "data"),  # NOTE: This intentionally uses index="psf".
    State("depth-step-size", "value"),
    State(dict(type="color-scale", index=MATCH), "value"),
    State(dict(type="image-slider", index=MATCH), "id"),
    prevent_initial_call=True,
)
def update_image_from_slider(
    value: int, tooltip: dict, n_frames: int, step_size: float, color_scale: str, id: dict[str, str]
):
    """Re-plot image based on frame/depth selected from slider."""
    global image_data
    # tooltip.transform (clientside js )would render faster than updating the template in a server side callback,
    # however, it only takes a single input ``value`` and there doesn't seem a trivial way to also pass ``step_size``
    # and ``n_frames`` to the js transform function (and have it dynamically served/reloaded).
    tooltip["template"] = slider_template_str(value, _center_frame(n_frames), step_size)
    fig = plot_image(image_data[id["index"]][value, :, :], color_scale=color_scale)
    return fig, tooltip


@callback(
    Output({"type": "image-slider", "index": ALL}, "tooltip", allow_duplicate=True),
    Input("depth-step-size", "value"),
    State({"type": "image-slider", "index": ALL}, "value"),
    State({"type": "image-slider", "index": ALL}, "tooltip"),
    State({"type": "n-frames", "index": "psf"}, "data"),
    prevent_initial_call=True,
)
def update_slider_tooltip(depth_step_size: float, values: list[int], tooltips: dict, n_frames: int):
    for i, value in enumerate(values):
        tooltips[i]["template"] = slider_template_str(value, _center_frame(n_frames), depth_step_size)
    return tooltips


def _display_image_with_slider(data: ArrayLike, id: str):
    """Plot image and display frame/depth slider (where appropriate)."""
    if data is None:
        return no_update

    n_frames = len(data)
    center_frame = _center_frame(n_frames)
    default_slider_value = center_frame
    fig = plot_image(data[center_frame, :, :])
    children = [
        dcc.Graph(id=dict(type="image-graph", index=id), figure=fig, responsive=True, style={"height": "100vh"}),
        html.Label("Select Image Frame/Depth"),
        dcc.Slider(
            0,
            len(data) - 1,
            step=1,
            value=default_slider_value,
            id=dict(type="image-slider", index=id),
            disabled=n_frames <= 1,
            tooltip={
                "always_visible": n_frames > 1,
                "template": slider_template_str(default_slider_value, center_frame, app_settings.DEPTH_STEP_SIZE),
            },
        ),
        html.Label("Color Scale"),
        dcc.Dropdown(
            id=dict(type="color-scale", index=id),
            options=colorscales,
            value=app_settings.IMSHOW_COLOR_SCALE,
        ),
    ]
    return children, n_frames


def upload_file(contents: str, filename: str):
    """Read in data/image files."""
    if contents is not None:
        logger.info(f"Uploading {filename}...")
        _content_type, content_string = contents.split(",", maxsplit=1)
        decoded = base64.b64decode(content_string)
        bytestream = io.BytesIO(decoded)
        data = flfm_io.open(bytestream)
        logger.info(f"Uploading {filename} completed.")
        return data


@callback(
    Output(dict(type="display", index="reconstruction"), "children", allow_duplicate=True),
    Input("run-reconstruction", "n_clicks"),
    Input("rl-iters", "value"),
    State("center_x", "value"),
    State("center_y", "value"),
    State("radius", "value"),
    prevent_initial_call=True,
)
def reconstruct(n_clicks: int, rl_iters: int, center_x: int, center_y: int, radius: int):
    """RUN FLFM reconstruction and return reconstructed image."""
    global image_data
    if n_clicks == 0:
        return no_update

    if image_data["psf"] is None or image_data["image"] is None:
        raise no_update

    id = "reconstruction"
    image_data["uncropped_reconstruction"] = flfm_restoration.richardson_lucy(
        image_data["image"], image_data["psf"], num_iter=rl_iters
    )
    image_data[id] = flfm.util.crop_and_apply_circle_mask(
        image_data["uncropped_reconstruction"],
        center=(center_x, center_y),
        radius=radius,
    )
    children, _n_frames = _display_image_with_slider(image_data[id], id=id)
    return children


@callback(
    Output(dict(type="image-graph", index="psf"), "figure", allow_duplicate=True),
    Output("normalize-psf", "disabled"),
    Input("normalize-psf", "n_clicks"),
    State(dict(type="image-slider", index="psf"), "value"),
    State(dict(type="color-scale", index="psf"), "value"),
    prevent_initial_call=True,
)
def normalize_psf(n_clicks: int, frame: int, color_scale: str):
    """Normalize PSF and disable button once normalized."""
    global image_data
    if n_clicks == 0 or image_data["psf"] is None:
        return no_update

    id = "psf"
    image_data[id] = image_data[id] / psf_sum(image_data[id])
    fig = plot_image(image_data[id][frame, :, :], color_scale=color_scale)
    return fig, True


@callback(
    Output(dict(type="image-graph", index="reconstruction"), "figure", allow_duplicate=True),
    Input("center_x", "value"),
    Input("center_y", "value"),
    Input("radius", "value"),
    State(dict(type="image-slider", index="reconstruction"), "value"),
    prevent_initial_call=True,
)
def update_crop(center_x: int, center_y: int, radius: int, frame: int):
    if image_data["uncropped_reconstruction"] is None:
        return no_update

    image_data["reconstruction"] = flfm.util.crop_and_apply_circle_mask(
        image_data["uncropped_reconstruction"], center=(center_x, center_y), radius=radius
    )
    fig = plot_image(image_data["reconstruction"][frame, :, :])
    return fig


@callback(
    Output(dict(type="image-graph", index=MATCH), "figure", allow_duplicate=True),
    Input(dict(type="color-scale", index=MATCH), "value"),
    State(dict(type="image-slider", index=MATCH), "value"),
    State(dict(type="color-scale", index=MATCH), "id"),
    prevent_initial_call=True,
)
def update_color_scale(color_scale: str, frame: int, id: str):
    fig = plot_image(image_data[id["index"]][frame, :, :], color_scale=color_scale)
    return fig


def start_app():
    match app_settings.WEB_API:
        case "flask":
            if not app_settings.DEBUG:
                raise NotImplementedError("The flask server is only suitable for DEBUG mode.")
            dash_app.run(host=app_settings.HOST, port=app_settings.PORT, debug=app_settings.DEBUG)
        case "fastapi":
            import uvicorn

            uvicorn.run(app, host=app_settings.HOST, port=app_settings.PORT, log_level=settings.LOG_LEVEL.lower())
        case _:
            raise NotImplementedError()


if __name__ == "__main__":
    start_app()
