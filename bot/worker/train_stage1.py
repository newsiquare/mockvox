import os, traceback
import numpy as np
from scipy.io import wavfile
from bot.config import get_config, celery_config
from bot.core import Slicer, load_audio
from .worker import celeryApp
from bot.utils import BotLogger

cfg = get_config()
UPLOAD_PATH = cfg.UPLOAD_PATH
os.makedirs(cfg.SLICED_ROOT_PATH, exist_ok=True)

@celeryApp.task(name="train_stage1", bind=True)
def process_file_task(self, file_name: str):
    try:
        stem, _ = os.path.splitext(file_name)
        file_path = os.path.join(UPLOAD_PATH, file_name)
        
        # 文件切割
        sliced_path = os.path.join(
            cfg.SLICED_ROOT_PATH, 
            stem
        )
        os.makedirs(sliced_path, exist_ok=True)
        
        slicer = Slicer(
            sr=32000,  # 长音频采样率
            threshold=      int(cfg.THRESHOLD),     # 音量小于这个值视作静音的备选切割点
            min_length=     int(cfg.MIN_LENGTH),    # 每段最小多长，如果第一段太短一直和后面段连起来直到超过这个值
            min_interval=   int(cfg.MIN_INTERVAL),  # 最短切割间隔
            hop_size=       int(cfg.HOP_SIZE),      # 怎么算音量曲线，越小精度越大计算量越高（不是精度越大效果越好）
            max_sil_kept=   int(cfg.MAX_SIL_KEPT),  # 切完后静音最多留多长
        )

        try:
            audio = load_audio(file_path, 32000)

            for chunk, start, end in slicer.slice(audio):  # start和end是帧数
                tmp_max = np.abs(chunk).max()
                if(tmp_max>1):chunk/=tmp_max
                chunk = (chunk / tmp_max * (cfg.MAX_NORMALIZED * cfg.ALPHA_MIX)) + (1 - cfg.ALPHA_MIX) * chunk
                sliced_file = os.path.join(sliced_path, "%010d_%010d.wav" % (start, end))
                wavfile.write(
                    sliced_file,
                    32000,
                    # chunk.astype(np.float32),
                    (chunk * 32767).astype(np.int16),
                )

        except:
            BotLogger.error(f"文件切割异常 | {file_path} | {traceback.format_exc()}")

        BotLogger.info(
            "文件已切割",
            extra={
                "action": "file_sliced",
                "task_id": self.request.id,
                "path": sliced_path
            }
        )
        
        return {"status": "success", "path": sliced_path}
    
    except Exception as e:
        raise self.retry(exc=e, countdown=60, max_retries=3)
    