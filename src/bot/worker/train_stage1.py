import os, traceback
import numpy as np
import time
from scipy.io import wavfile
from bot.config import get_config, PRETRAINED_PATH, UPLOAD_PATH, SLICED_ROOT_PATH, DENOISED_ROOT_PATH, ASR_PATH
from bot.core import Slicer, load_audio, AudioDenoiser, AutoSpeechRecognition
from .worker import celeryApp
from bot.utils import BotLogger
from typing import List
from pathlib import Path

cfg = get_config()
os.makedirs(SLICED_ROOT_PATH, exist_ok=True)

@celeryApp.task(name="train_stage1", bind=True)
def process_file_task(self, file_name: str, ifDenoise: bool):
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
        if(ifDenoise):
            asr_results = batch_asr(denoised_files, asr_path)
            path_result = denoised_path
        else:
            asr_results = batch_asr(sliced_files, asr_path)
            path_result = sliced_path

        BotLogger.info(
            "语音已识别",
            extra={
                "action": "asr",
                "task_id": self.request.id,
                "path": asr_path
            }
        )
        
        return {
            "status": "success", 
            "results": asr_results, 
            "path": Path(path_result).name,
            "time":time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        }
    
    except Exception as e:
        BotLogger.error(
            f"任务失败 | 文件: {file_name} | 错误跟踪:\n{traceback.format_exc()}"
        )
        raise self.retry(exc=e, countdown=60, max_retries=3)

def slice_audio(input_path: str, output_dir: str) -> List[str]:
    """音频文件切割函数
    
    Args:
        input_path: 输入音频文件路径
        output_dir: 切片输出目录
        
    Returns:
        切片输出文件名(数组)
        
    Raises:
        FileNotFoundError: 文件不存在
        RuntimeError: 切割处理失败
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"输入文件不存在: {input_path}")
    
    try:
        slicer = Slicer(
            sr=32000,                               # 长音频采样率
            threshold=      int(cfg.THRESHOLD),     # 音量小于这个值视作静音的备选切割点
            min_length=     int(cfg.MIN_LENGTH),    # 每段最小多长，如果第一段太短一直和后面段连起来直到超过这个值
            min_interval=   int(cfg.MIN_INTERVAL),  # 最短切割间隔
            hop_size=       int(cfg.HOP_SIZE),      # 怎么算音量曲线，越小精度越大计算量越高（不是精度越大效果越好）
            max_sil_kept=   int(cfg.MAX_SIL_KEPT),  # 切完后静音最多留多长
        )

        audio = load_audio(input_path, 32000)
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        sliced_files = []
        for chunk, start, end in slicer.slice(audio):
            # 音量归一化处理
            tmp_max = np.abs(chunk).max()
            if(tmp_max>1):chunk/=tmp_max
            chunk = (chunk / tmp_max * (cfg.MAX_NORMALIZED * cfg.ALPHA_MIX)) + (1 - cfg.ALPHA_MIX) * chunk

            sliced_file = os.path.join(
                output_dir,
                f"{start:010d}_{end:010d}.wav"  
            )
            sliced_files.append(sliced_file)
            
            wavfile.write(
                sliced_file,
                32000,
                (chunk * 32767).astype(np.int16)
            )

        return sliced_files

    except Exception as e:
        BotLogger.error(
            f"切割异常 | 文件: {input_path} | 错误: {str(e)}",
            extra={"action": "slice_error"}
        )
        raise RuntimeError(f"音频切割失败: {str(e)}") from e

def batch_denoise(file_list: List[str], output_dir: str) -> List[str]:
    """批量降噪函数
    
    Args:
        file_list: 切片文件名数组
        output_dir: 降噪输出目录
        
    Returns:
        降噪输出文件名(数组)
        
    Raises:
        RuntimeError: 降噪处理失败
    """
    try:
        denoise_model = os.path.join(PRETRAINED_PATH, 'damo/speech_frcrn_ans_cirm_16k')
        denoise_model = denoise_model if os.path.exists(denoise_model) else 'damo/speech_frcrn_ans_cirm_16k'
        denoiser = AudioDenoiser(model_name=denoise_model)     
        Path(output_dir).mkdir(parents=True, exist_ok=True)   

        denoised_files = []        
        for file in file_list:
            denoised_file = denoiser.denoise(file, output_dir=output_dir)
            denoised_files.append(denoised_file)
        return denoised_files
    
    except Exception as e:
        BotLogger.error(
            f"降噪异常 | 路径: {output_dir} | 错误: {str(e)}",
            extra={"action": "denoise_error"}
        )
        raise RuntimeError(f"降噪失败: {str(e)}") from e

def batch_asr(file_list: List[str], output_dir: str):
    """批量识别函数
    
    Args:
        file_list: 降噪文件名数组
        output_dir: 语音识别输出目录
        
    Returns:
        语音识别结果(数组)
        
    Raises:
        RuntimeError: 识别处理失败
    """
    try:
        asr_model = os.path.join(PRETRAINED_PATH, 'iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch')
        asr_model = asr_model if os.path.exists(asr_model) else 'iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch'
        punc_model = os.path.join(PRETRAINED_PATH, 'iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch')
        punc_model = punc_model if os.path.exists(punc_model) else 'iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch'

        asr = AutoSpeechRecognition(asr_model_name=asr_model, punc_model_name=punc_model)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_file = os.path.join(output_dir, "output.txt")

        results = []
        with open(output_file, 'w', encoding='utf-8') as f:
            for file in file_list:
                result = asr.speech_recognition(input_path=file)
                f.writelines(f"{item}\n" for item in result)
                results.extend(result)
        return results

    except Exception as e:
        BotLogger.error(
            f"语音识别异常 | 路径: {output_dir} | 错误: {str(e)}",
            extra={"action": "asr_error"}
        )
        raise RuntimeError(f"语音识别失败: {str(e)}") from e
