# -*- coding: utf-8 -*-
import os, traceback
import time
from pathlib import Path
from typing import Optional
from collections import OrderedDict

from bot.config import get_config, UPLOAD_PATH, SLICED_ROOT_PATH, DENOISED_ROOT_PATH, ASR_PATH
from bot.engine.v2 import slice_audio, batch_denoise, batch_asr as batch_asr_v2
from bot.engine.v4 import batch_asr as batch_asr_v4
from .worker import celeryApp
from bot.utils import BotLogger

cfg = get_config()
os.makedirs(SLICED_ROOT_PATH, exist_ok=True)

@celeryApp.task(name="train_stage1", bind=True)
def process_file_task(
    self, 
    file_name: str,
    version: Optional[str] = 'v4',
    language: Optional[str] = 'zh',
    region: Optional[str] = None, 
    ifDenoise: Optional[bool] = True
):
    try:
        stem, _ = os.path.splitext(file_name)
        file_path = os.path.join(UPLOAD_PATH, file_name)
        
        # 文件切割
        sliced_path = os.path.join(SLICED_ROOT_PATH, stem)
        sliced_files = slice_audio(file_path, sliced_path)

        BotLogger.info(
            "文件已切割",
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
                "已降噪",
                extra={
                    "action": "file_denoised",
                    "task_id": self.request.id,
                    "path": denoised_path
                }
            )

        # 语音识别
        asr_path = os.path.join(ASR_PATH, stem)
        if(version=='v4'):
            if(ifDenoise):
                asr_results = batch_asr_v4(language, denoised_files, asr_path)
                path_result = denoised_path
            else:
                asr_results = batch_asr_v4(language, sliced_files, asr_path)
                path_result = sliced_path
        elif(version=='v2'):
            if(ifDenoise):
                asr_results = batch_asr_v2(language, denoised_files, asr_path)
                path_result = denoised_path
            else:
                asr_results = batch_asr_v2(language, sliced_files, asr_path)
                path_result = sliced_path
        else:
            BotLogger.error(f"不支持的版本 | 文件: {file_name} | 错误跟踪:\n{traceback.format_exc()}")

        BotLogger.info(
            "语音已识别",
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
            f"任务失败 | 文件: {file_name} | 错误跟踪:\n{traceback.format_exc()}"
        )
        raise self.retry(exc=e, countdown=60, max_retries=3)