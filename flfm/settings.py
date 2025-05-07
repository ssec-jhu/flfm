from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FLFM_", case_sensitive=True)
    BACKEND: Literal["jax", "torch"] = "torch"
    DEFAULT_RL_ITERS: int = 10
    DEFAULT_CENTER_X: int = 1000
    DEFAULT_CENTER_Y: int = 980
    DEFAULT_RADIUS: int = 230


settings = Settings()
