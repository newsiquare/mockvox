from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    # Global
    ENV: str = Field("dev", env="ENV")
    NAME: str = Field("bot", env="NAME")
    VERSION: str = Field("0.0.1", env="VERSION")

    # 上传文件大小限制10M
    MAX_UPLOAD_SIZE = int = Field(10, env="MAX_UPLOAD_SIZE")

    # Slice
    THRESHOLD: int = Field(-34, env="THRESHOLD")
    MIN_LENGTH: int = Field(4000, env="MIN_LENGTH")
    MIN_INTERVAL: int = Field(300, env="MIN_INTERVAL")
    HOP_SIZE: int = Field(10, env="HOP_SIZE")
    MAX_SIL_KEPT: int = Field(500, env="MAX_SIL_KEPT")

    # Security
    SECRET_KEY: str = Field(..., env="SECRET_KEY")
    SECRET_KEY_EXPIRE_MINUTES: int = Field(1440, env="SECRET_KEY_EXPIRE_MINUTES")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True  # 环境变量大小写敏感

def get_settings():
    return Settings()