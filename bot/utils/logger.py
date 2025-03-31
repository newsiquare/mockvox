import logging
from bot.config import BASE_DIR
import os

log_file = os.path.join(BASE_DIR, "api.log")

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file)
    ]
)
BotLogger = logging.getLogger(__name__)