import logging
from logging.handlers import RotatingFileHandler
from bot.config import BASE_DIR
import os

class ConditionalFormatter(logging.Formatter):
    def format(self, record):
        if hasattr(record, 'action'):
            if record.action == "file_saved":
                # 动态生成日志格式
                self._style._fmt = '%(asctime)s - %(name)s - %(levelname)s - ' \
                    '[action=%(action)s] [filename=%(filename)s] ' \
                    '[filesize=%(filesize)s] [content_type=%(content_type)s] - %(message)s'
        else:
            # 默认格式
            self._style._fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        return super().format(record)

log_dir = os.path.join(BASE_DIR, "logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "bot.log")

BotLogger = logging.getLogger("BotLog")
BotLogger.setLevel(logging.INFO)

if not BotLogger.handlers:
    # 定义日志格式
    formatter = ConditionalFormatter(
        datefmt='%Y-%m-%d %H:%M:%S %Z'
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = RotatingFileHandler(
        filename=log_file,
        maxBytes=5 * 1024 * 1024,  # 5MB
        backupCount=10,
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)

    BotLogger.addHandler(console_handler)
    BotLogger.addHandler(file_handler)

    BotLogger.propagate = False