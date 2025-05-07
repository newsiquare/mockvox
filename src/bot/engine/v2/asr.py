# -*- coding: utf-8 -*-
"""
语音识别(Auto Speech Recognition, asr)模块
"""
from funasr import AutoModel
from typing import Optional, List, Union
from pathlib import Path
import torch
import os
import gc
import json

from bot.config import PRETRAINED_PATH
from bot.utils import BotLogger

class AutoSpeechRecognition:
    def __init__(self,
                 language: str = 'zh',
                 asr_model_name: str = 'iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch',
                 vad_model_name: str = 'iic/speech_fsmn_vad_zh-cn-16k-common-pytorch',
                 punc_model_name: str = 'iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch',
                 device: Optional[str] = None
        ): 
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # 语音识别
        self.model = AutoModel(
            model=os.path.join(PRETRAINED_PATH,asr_model_name), model_revision='v2.0.4',
            vad_model=os.path.join(PRETRAINED_PATH,vad_model_name), vad_model_revision='v2.0.4',
            punc_model=os.path.join(PRETRAINED_PATH,punc_model_name), punc_model_revision='v2.0.4',
            device=self.device,
            disable_update=True
        )
        
    def speech_recognition(self, input_path: str) -> List:
        try:
            asr_result = self.model.generate(input=input_path)
            if not isinstance(asr_result, list) or len(asr_result) == 0:
                raise ValueError("ASR结果必须是包含至少一个元素的列表")

        except Exception as e:
            raise RuntimeError(f"语音识别&标点恢复失败: {str(e)}") from e

        return asr_result
    
def load_asr_data(asr_dir: Union[str,Path]) -> List[dict]:
    """
    解析ASR识别结果文件
    返回格式: [{"key": "文件名", "text": "识别文本"}, ...]
    """
    result = []
    asr_file = Path(asr_dir) / 'output.json'
    try:
        with open(asr_file, 'r', encoding='utf-8') as f:
            result = json.load(f)
    except (SyntaxError, ValueError) as e:
        BotLogger.error(f"ASR文件格式错误: {str(e)}")
    except FileNotFoundError:
        BotLogger.error(f"ASR文件不存在: {asr_file}")
    return result

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
        output_file = os.path.join(output_dir, "output.json")

        results = []
        for file in file_list:
            result = asr.speech_recognition(input_path=file)
            results.extend(result)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        del asr
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.ipc_collect()
            gc.collect()
        return results

    except Exception as e:
        BotLogger.error(
            f"语音识别异常 | 路径: {output_dir} | 错误: {str(e)}",
            extra={"action": "asr_error"}
        )
        raise RuntimeError(f"语音识别失败: {str(e)}") from e

if __name__ == '__main__':
    # 示例用法
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, help='processed file name.')
    parser.add_argument('--no-denoised', action='store_false', dest='denoised',  # 添加反向参数
                        help='disable denoised processing (default: enable denoised)')
    parser.set_defaults(denoised=True)

    args = parser.parse_args()

    asr = AutoSpeechRecognition()

    from bot.config import ASR_PATH, DENOISED_ROOT_PATH, SLICED_ROOT_PATH
    import os

    asr_path = os.path.join(ASR_PATH, args.file)
    Path(asr_path).mkdir(parents=True, exist_ok=True)
    output_file = os.path.join(asr_path, "output.json")

    if args.denoised:
        root_dir = os.path.join(DENOISED_ROOT_PATH, args.file)
    else:
        root_dir = os.path.join(SLICED_ROOT_PATH, args.file)

    file_list = [
        os.path.join(root_dir, f) 
        for f in os.listdir(root_dir)
        if os.path.isfile(os.path.join(root_dir, f))  # 过滤掉目录
    ]
    results = []
    for file in file_list:
        result = asr.speech_recognition(input_path=file)
        results.extend(result)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
