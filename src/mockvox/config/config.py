# config.py
import os
from pathlib import Path
from dotenv import load_dotenv

BASE_PATH = Path(__file__).resolve().parent.parent.parent.parent
DATA_PATH = os.path.join(BASE_PATH, "data")
PRETRAINED_PATH = os.path.join(BASE_PATH, "pretrained")
LOG_PATH = os.path.join(BASE_PATH, "log")
UPLOAD_PATH = os.path.join(DATA_PATH, "upload")
SLICED_ROOT_PATH = os.path.join(DATA_PATH, "sliced")
DENOISED_ROOT_PATH = os.path.join(DATA_PATH, "denoised")
ASR_PATH = os.path.join(DATA_PATH, "asr")
PROCESS_PATH = os.path.join(DATA_PATH, "process")
WEIGHTS_PATH = os.path.join(DATA_PATH, "weights")
OUT_PUT_PATH = os.path.join(DATA_PATH, "output")
REF_AUDIO_PATH = os.path.join(DATA_PATH, "refAudio")

SOVITS_MODEL_CONFIG = os.path.join(BASE_PATH, "src/mockvox/config/s2.json")
GPT_MODEL_CONFIG = os.path.join(BASE_PATH, "src/mockvox/config/s1.json")

PRETRAINED_S2G_FILE = os.path.join(PRETRAINED_PATH, 'GPT-SoVITS/gsv-v2final-pretrained/s2G2333k.pth')
PRETRAINED_S2D_FILE = os.path.join(PRETRAINED_PATH, 'GPT-SoVITS/gsv-v2final-pretrained/s2D2333k.pth')
PRETRAINED_GPT_FILE = os.path.join(PRETRAINED_PATH, 'GPT-SoVITS/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt')
PRETRAINED_S2GV4_FILE = os.path.join(PRETRAINED_PATH, 'GPT-SoVITS/gsv-v4-pretrained/s2Gv4.pth')
PRETRAINED_VOCODER_FILE = os.path.join(PRETRAINED_PATH, 'GPT-SoVITS/gsv-v4-pretrained/vocoder.pth')
PRETRAINED_T2SV4_FILE = os.path.join(PRETRAINED_PATH, 'GPT-SoVITS/s1v3.ckpt')

SOVITS_G_WEIGHTS_FILE = 'gen.pth'
SOVITS_D_WEIGHTS_FILE = 'disc.pth'
SOVITS_HALF_WEIGHTS_FILE = 'sovits.pth'
GPT_WEIGHTS_FILE = 'decoder.pth'
GPT_HALF_WEIGHTS_FILE = 'gpt.pth'
OUT_PUT_FILE = 'output'

load_dotenv(BASE_PATH / ".env", override=True)  # 加载.env文件

class Settings:
    # Global
    ENV: str = os.environ.get("ENV", "dev")
    NAME: str = os.environ.get("NAME", "mockvox")
    VERSION: str = os.environ.get("VERSION", "0.0.1")
    MAIN_HOST: str = os.environ.get("MAIN_HOST", "0.0.0.0")
    MAIN_PORT: int = int(os.environ.get("MAIN_PORT", "5000"))  # 端口号转int

    # 上传文件
    MAX_UPLOAD_SIZE: int = int(os.environ.get("MAX_UPLOAD_SIZE", "10"))*1024*1024 # (单位：MB)

    # Slice 配置
    THRESHOLD: int = int(os.environ.get("THRESHOLD", "-34"))
    MIN_LENGTH: int = int(os.environ.get("MIN_LENGTH", "4000"))
    MIN_INTERVAL: int = int(os.environ.get("MIN_INTERVAL", "300"))
    HOP_SIZE: int = int(os.environ.get("HOP_SIZE", "10"))
    MAX_SIL_KEPT: int = int(os.environ.get("MAX_SIL_KEPT", "500"))
    MAX_NORMALIZED: float = float(os.environ.get("MAX_NORMALIZED","0.9"))
    ALPHA_MIX: float = float(os.environ.get("ALPHA_MIX","0.25"))

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