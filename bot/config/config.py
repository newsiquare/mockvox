# config.py
import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")  # 加载.env文件

class Settings:
    # Global
    ENV: str = os.environ.get("ENV", "dev")
    NAME: str = os.environ.get("NAME", "bot")
    VERSION: str = os.environ.get("VERSION", "0.0.1")
    MAIN_HOST: str = os.environ.get("MAIN_HOST", "0.0.0.0")
    MAIN_PORT: int = int(os.environ.get("MAIN_PORT", "5000"))  # 端口号转int

    # 上传文件
    MAX_UPLOAD_SIZE: int = int(os.environ.get("MAX_UPLOAD_SIZE", "10")) # (单位：MB)
    UPLOAD_PATH: str = os.environ.get("UPLOAD_PATH","data/uploads")

    # Slice 配置（全部转为int类型）
    THRESHOLD: int = int(os.environ.get("THRESHOLD", "-34"))
    MIN_LENGTH: int = int(os.environ.get("MIN_LENGTH", "4000"))
    MIN_INTERVAL: int = int(os.environ.get("MIN_INTERVAL", "300"))
    HOP_SIZE: int = int(os.environ.get("HOP_SIZE", "10"))
    MAX_SIL_KEPT: int = int(os.environ.get("MAX_SIL_KEPT", "500"))

    SLICED_ROOT_PATH: str = os.environ.get("SLICED_ROOT_PATH","data/sliced")

    # Redis 配置
    REDIS_HOST: str = os.environ.get("REDIS_HOST", "127.0.0.1")
    REDIS_PORT: str = os.environ.get("REDIS_PORT", "6379")
    REDIS_DB_BROKER: str = os.environ.get("REDIS_DB_BROKER","0")
    REDIS_DB_RESULT: str = os.environ.get("REDIS_DB_RESULT","1")
    REDIS_MEMORY_LIMIT: str = os.environ.get("REDIS_MEMORY_LIMIT","2GB")
    REDIS_PASSWORD: str = os.environ.get("REDIS_PASSWORD")

    # Security
    SECRET_KEY: str = os.environ.get("SECRET_KEY")
    SECRET_KEY_EXPIRE_MINUTES: int = int(os.environ.get("SECRET_KEY_EXPIRE_MINUTES", "1440"))

def get_config():
    return Settings()