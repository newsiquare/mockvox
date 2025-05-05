# -*- coding: utf-8 -*-
"""
语音识别(Auto Speech Recognition, asr)模块
"""
from funasr import AutoModel
from typing import Optional, List, Union
from pathlib import Path
import torch
import json

from bot.utils import BotLogger

funasr_models = {}

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
        if language in funasr_models:
            self.model = funasr_models[language]
        else:
            self.model = AutoModel(
                model=asr_model_name, model_revision='v2.0.4',
                vad_model=vad_model_name, vad_model_revision='v2.0.4',
                punc_model=punc_model_name, punc_model_revision='v2.0.4',
                device=self.device,
                disable_update=True
            )

            funasr_models[language]=self.model
        
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
        print(file)
        result = asr.speech_recognition(input_path=file)
        results.extend(result)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
