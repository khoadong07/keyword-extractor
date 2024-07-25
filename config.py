import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    LOG_PATH: str
    FIREWORKS_URL: str
    FIREWORKS_TOKEN: str
    FIREWORKS_API_MAX_TOKEN: str
    TEMPERATURE: str
    FIREWORKS_MODEL: str

    class Config:
        env_file = ".env"


settings = Settings()
