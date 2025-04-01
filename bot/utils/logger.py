import logging
from logging.handlers import RotatingFileHandler
from bot.config import BASE_DIR
import os

class ConditionalFormatter(logging.Formatter):
    _FORMATS = {
        "file_saved": (
            '%(asctime)s - %(name)s - %(levelname)s - '
            '[action=%(action)s] [file_name=%(file_name)s] '
            '[file_size=%(file_size)s] [content_type=%(content_type)s] - %(message)s'
        ),
        "task_submitted": (
            '%(asctime)s - %(name)s - %(levelname)s - '
            '[action=%(action)s] [task_id=%(task_id)s] '
            '[file_name=%(file_name)s] - %(message)s'
        ),
        "default": '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    }

    def format(self, record):        
        fmt_template = self._FORMATS.get(
            getattr(record, 'action', None), 
            self._FORMATS['default']
        )
        
        formatter = logging.Formatter(
            fmt=fmt_template,
            datefmt=self.datefmt
        )
        return formatter.format(record)

log_dir = os.path.join(BASE_DIR, "logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "bot.log")

BotLogger = logging.getLogger("BotLog")
BotLogger.setLevel(logging.INFO)

if not BotLogger.handlers:
    formatter = ConditionalFormatter(
        datefmt='%Y-%m-%d %H:%M:%S %Z'
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = RotatingFileHandler(
        filename=log_file,
        maxBytes=5 * 1024 * 1024,  # 5MB日志文件限制，循环10个文件
        backupCount=10,
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)

    BotLogger.addHandler(console_handler)
    BotLogger.addHandler(file_handler)

    BotLogger.propagate = False