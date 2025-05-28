import logging
from logging.handlers import RotatingFileHandler
from mockvox.config import LOG_PATH
import os

class ConditionalFormatter(logging.Formatter):
    _FORMATS = {
        "file_saved": (
            '%(asctime)s - %(name)s - %(levelname)s - '
            '[action=%(action)s] [file_id=%(file_id)s] '
            '[file_size=%(file_size)s] [content_type=%(content_type)s] - %(message)s'
        ),
        "stage1_task_submitted": (
            '%(asctime)s - %(name)s - %(levelname)s - '
            '[action=%(action)s] [task_id=%(task_id)s] '
            '[file_id=%(file_id)s] - %(message)s'
        ),
        "stage2_task_submitted": (
            '%(asctime)s - %(name)s - %(levelname)s - '
            '[action=%(action)s] [task_id=%(task_id)s] '
            '[file_id=%(file_id)s] - %(message)s'
        ),
        "file_sliced": (
            '%(asctime)s - %(name)s - %(levelname)s - '
            '[action=%(action)s] [task_id=%(task_id)s] '
            '[path=%(path)s] - %(message)s'
        ),
        "file_denoised": (
            '%(asctime)s - %(name)s - %(levelname)s - '
            '[action=%(action)s] [task_id=%(task_id)s] '
            '[path=%(path)s] - %(message)s'
        ),
        "asr": (
            '%(asctime)s - %(name)s - %(levelname)s - '
            '[action=%(action)s] [task_id=%(task_id)s] '
            '[path=%(path)s] - %(message)s'
        ),
        "data_processed": (
            '%(asctime)s - %(name)s - %(levelname)s - '
            '[action=%(action)s] [file_id=%(file_id)s]'
            '[json=%(json_file)s] - %(message)s'
        ),
        "feature_extracted": (
            '%(asctime)s - %(name)s - %(levelname)s - '
            '[action=%(action)s] [file_id=%(file_id)s]'
            ' - %(message)s'
        ),
        "text_to_semantic": (
            '%(asctime)s - %(name)s - %(levelname)s - '
            '[action=%(action)s] [file_id=%(file_id)s]'
            '[json=%(json_file)s] - %(message)s'
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

os.makedirs(LOG_PATH, exist_ok=True)
log_file = os.path.join(LOG_PATH, "mockvox.log")

MockVoxLogger = logging.getLogger("MockVox")
MockVoxLogger.setLevel(logging.INFO)

if not MockVoxLogger.handlers:
    formatter = ConditionalFormatter(
        datefmt='%Y-%m-%d %H:%M:%S %Z'
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = RotatingFileHandler(
        filename=log_file,
        maxBytes=5 * 1024 * 1024,  # 5MB 日志文件限制，循环10个文件
        backupCount=10,
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)

    MockVoxLogger.addHandler(console_handler)
    MockVoxLogger.addHandler(file_handler)

    MockVoxLogger.propagate = False # 禁止向上传递