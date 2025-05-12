from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

_log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FLFM_", case_sensitive=True)
    BACKEND: Literal["jax", "torch"] = "torch"
    DEFAULT_RL_ITERS: int = 10
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = _log_format
    DEFAULT_CENTER_X: int = 1000
    DEFAULT_CENTER_Y: int = 980
    DEFAULT_RADIUS: int = 230


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FLFM_APP_", case_sensitive=True)
    HOST: str = "127.0.0.1"
    PORT: int = 8080
    LOADING_TYPE: Literal["graph", "cube", "circle", "dot", "default"] = "default"
    LOADING_DELAY_SHOW: int = 1e3  # (ms) See https://dash.plotly.com/dash-core-components/loading.
    DEPTH_STEP_SIZE: float = 5
    DEPTH_UNIT: str = "µm"
    DEBUG: bool = False
    WEB_API: Literal["fastapi", "flask"] = "fastapi"
    RUN_DUMMY_RECONSTRUCTION_AT_STARTUP: bool = False  # Incur jax/pytorch jit overhead at startup (fastapi only).
    STARTUP_DUMMY_RECONSTRUCTION_N_ITERS: int = 1
    DEBOUNCE: bool = True  # See https://dash.plotly.com/dash-core-components/input.
    THEME: str = "YETI"  # See https://dash-bootstrap-components.opensource.faculty.ai/docs/themes/explorer/.
    IMSHOW_COLOR_SCALE: str = "gray"  # NOTE: Full list presented in app dropdown menu.
    IMSHOW_SHOW_TICK_LABELS: bool = False
    DUMMY_PSF_FILEPATH: Path | None = None
    DUMMY_LIGHT_FILED_IMAGE_FILEPATH: Path | None = None


settings = Settings()
app_settings = AppSettings()
