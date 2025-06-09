# -*- coding: utf-8 -*-
import os, traceback
import time
from pathlib import Path
from typing import Optional
from collections import OrderedDict

from mockvox.config import get_config, UPLOAD_PATH, SLICED_ROOT_PATH, DENOISED_ROOT_PATH, ASR_PATH
from mockvox.engine.v2 import slice_audio, batch_denoise, batch_asr, load_asr_data, batch_add_asr
from .worker import celeryApp
from mockvox.utils import MockVoxLogger

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

        MockVoxLogger.info(
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
        
            MockVoxLogger.info(
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
            asr_results = batch_asr(language, ifDenoise, denoised_files, asr_path)
        else:
            asr_results = batch_asr(language, ifDenoise, sliced_files, asr_path)

        MockVoxLogger.info(
            "ASR done",
            extra={
                "action": "asr",
                "task_id": self.request.id,
                "path": asr_path
            }
        )

        results = OrderedDict()
        results["asr"] = asr_results
        results["file_id"] = stem
        
        return {
            "status": "success", 
            "results": results, 
            "time":time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        }
    
    except Exception as e:
        MockVoxLogger.error(
            f"Task failed: {file_name} \n\
                Traceback:\n{traceback.format_exc()}"
        )
        return {
            "status": "fail", 
            "results": {}, 
            "time":time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        }
    
@celeryApp.task(name="add audio", bind=True)
def add_audio_task(
    self, 
    file_id: str,
    file_name: str
):
    try:
        # 从ASR结果中读取参数
        asr_path = os.path.join(ASR_PATH, file_id)
        asr_file = Path(asr_path) / "output.json"
        if not asr_file.exists(): 
            MockVoxLogger.error(
                f"ASR file not exist: {asr_file}"
            )
            return
        asr_data = load_asr_data(asr_path)

        file_path = os.path.join(UPLOAD_PATH, file_name)
        
        # 文件切割
        sliced_path = os.path.join(SLICED_ROOT_PATH, file_id)
        sliced_files = slice_audio(file_path, sliced_path)

        MockVoxLogger.info(
            "Audio sliced",
            extra={
                "action": "file_sliced",
                "task_id": self.request.id,
                "path": sliced_path
            }
        )

        # 降噪
        if(asr_data['denoised']):
            denoised_path = os.path.join(DENOISED_ROOT_PATH, file_id)
            denoised_files = batch_denoise(sliced_files, denoised_path)
        
            MockVoxLogger.info(
                "Audio files denoised",
                extra={
                    "action": "file_denoised",
                    "task_id": self.request.id,
                    "path": denoised_path
                }
            )

        # 语音识别
        if(asr_data['denoised']):
            asr_results = batch_add_asr(denoised_files, asr_data, asr_path)
        else:
            asr_results = batch_add_asr(sliced_files, asr_data, asr_path)

        MockVoxLogger.info(
            "ASR done",
            extra={
                "action": "asr",
                "task_id": self.request.id,
                "path": asr_path
            }
        )

        results = OrderedDict()
        results["asr"] = asr_results
        results["file_id"] = file_id
        
        return {
            "status": "success", 
            "results": results, 
            "time":time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        }
    
    except Exception as e:
        MockVoxLogger.error(
            f"Task failed: {file_name} \n\
                Traceback:\n{traceback.format_exc()}"
        )
        return {
            "status": "fail", 
            "results": {}, 
            "time":time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        }