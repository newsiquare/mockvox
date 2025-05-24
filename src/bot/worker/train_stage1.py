# -*- coding: utf-8 -*-
import os, traceback
import time
from pathlib import Path
from typing import Optional
from collections import OrderedDict

from bot.config import get_config, UPLOAD_PATH, SLICED_ROOT_PATH, DENOISED_ROOT_PATH, ASR_PATH
from bot.engine.v2 import slice_audio, batch_denoise, batch_asr
from .worker import celeryApp
from bot.utils import BotLogger

cfg = get_config()
os.makedirs(SLICED_ROOT_PATH, exist_ok=True)

@celeryApp.task(name="preprocess", bind=True)
def process_file_task(
    self, 
    file_name: str,
    language: Optional[str] = 'zh',
    ifDenoise: Optional[bool] = True
):
    try:
        stem, _ = os.path.splitext(file_name)
        file_path = os.path.join(UPLOAD_PATH, file_name)
        
        # 文件切割
        sliced_path = os.path.join(SLICED_ROOT_PATH, stem)
        sliced_files = slice_audio(file_path, sliced_path)

        BotLogger.info(
            "Audio sliced",
            extra={
                "action": "file_sliced",
                "task_id": self.request.id,
                "path": sliced_path
            }
        )

        # 降噪
        if(ifDenoise):
            denoised_path = os.path.join(DENOISED_ROOT_PATH, stem)
            denoised_files = batch_denoise(sliced_files, denoised_path)
        
            BotLogger.info(
                "Audio files denoised",
                extra={
                    "action": "file_denoised",
                    "task_id": self.request.id,
                    "path": denoised_path
                }
            )

        # 语音识别
        asr_path = os.path.join(ASR_PATH, stem)
        if(ifDenoise):
            asr_results = batch_asr(language, denoised_files, asr_path)
            path_result = denoised_path
        else:
            asr_results = batch_asr(language, sliced_files, asr_path)
            path_result = sliced_path

        BotLogger.info(
            "ASR done",
            extra={
                "action": "asr",
                "task_id": self.request.id,
                "path": asr_path
            }
        )

        results = OrderedDict()
        results["asr"] = asr_results
        results["path"] = Path(path_result).name
        
        return {
            "status": "success", 
            "results": results, 
            "time":time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        }
    
    except Exception as e:
        BotLogger.error(
            f"Task failed: {file_name} \n\
                Traceback:\n{traceback.format_exc()}"
        )
        return {
            "status": "fail", 
            "results": {}, 
            "time":time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        }