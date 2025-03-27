# config.py
import os
from dotenv import load_dotenv

load_dotenv()  # 加载 .env 文件到环境变量

class Settings:
    # Global
    ENV: str = os.environ.get("ENV", "dev")
    NAME: str = os.environ.get("NAME", "bot")
    VERSION: str = os.environ.get("VERSION", "0.0.1")
    MAIN_HOST: str = os.environ.get("MAIN_HOST", "0.0.0.0")
    MAIN_PORT: int = int(os.environ.get("MAIN_PORT", "5000"))  # 端口号转int

    # 上传文件大小限制（单位：MB）
    MAX_UPLOAD_SIZE: int = int(os.environ.get("MAX_UPLOAD_SIZE", "10"))

    # Slice 配置（全部转为int类型）
    THRESHOLD: int = int(os.environ.get("THRESHOLD", "-34"))
    MIN_LENGTH: int = int(os.environ.get("MIN_LENGTH", "4000"))
    MIN_INTERVAL: int = int(os.environ.get("MIN_INTERVAL", "300"))
    HOP_SIZE: int = int(os.environ.get("HOP_SIZE", "10"))
    MAX_SIL_KEPT: int = int(os.environ.get("MAX_SIL_KEPT", "500"))

    # Security（注意空字符串处理）
    SECRET_KEY: str = os.environ.get("SECRET_KEY", "")
    SECRET_KEY_EXPIRE_MINUTES: int = int(os.environ.get("SECRET_KEY_EXPIRE_MINUTES", "1440"))

def get_config():
    return Settings()