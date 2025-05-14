# -*- coding: utf-8 -*-
"""
语音识别(Auto Speech Recognition, asr)模块
"""
from typing import Optional, List, Union, Dict
from pathlib import Path
import torch
import os
import gc
import json

from funasr import AutoModel
from faster_whisper import WhisperModel

from bot.config import PRETRAINED_PATH
from bot.utils import BotLogger

class ChineseASR:
    def __init__(self,
                 language: str = "zh",  # 为了统一输入参数
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
        
    def execute(self, input_path: str):
        try:
            asr_result = self.model.generate(input=input_path)
            if not isinstance(asr_result, list) or len(asr_result) == 0:
                return None

        except Exception as e:
            raise RuntimeError(f"语音识别&标点恢复失败: {str(e)}") from e

        return asr_result, "zh"

class CantoneseASR:
    def __init__(self,
                 language: str = "can",
                 asr_model_name: str = 'iic/speech_UniASR_asr_2pass-cantonese-CHS-16k-common-vocab1468-tensorflow1-online',
                 device: Optional[str] = None
        ): 
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # 语音识别
        self.model = AutoModel(
            model=os.path.join(PRETRAINED_PATH,asr_model_name), model_revision='v2.0.4',
            vad_model=None, vad_model_revision=None,
            punc_model=None, punc_model_revision=None,
            device=self.device,
            disable_update=True
        )
        
    def execute(self, input_path: str):
        try:
            asr_result = self.model.generate(input=input_path)
            if not isinstance(asr_result, list) or len(asr_result) == 0:
                return None

        except Exception as e:
            raise RuntimeError(f"语音识别&标点恢复失败: {str(e)}") from e

        return asr_result, "can"

class FasterWhisperASR:
    def __init__(self,
        language: str = "auto",
        asr_model_name: str = 'faster-whisper-large-v3',
        device: Optional[str] = None
    ): 
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.language = language
        if self.language == "auto":
            self.language = None

        self.model = WhisperModel(
            os.path.join(PRETRAINED_PATH, asr_model_name), 
            device=self.device, 
            compute_type="default"
        )

    def execute(self, input_path: str):
        try:
            segments, info = self.model.transcribe(
                audio=input_path,
                beam_size=5,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=700),
                language=self.language
            )

            # if info.language == 'zh':
            #     asr = ChineseASR()
            #     asr_result = asr.execute(input_path)
            #     return asr_result, "zh"

            asr_result=[]
            for segment in segments:
                asr_result.extend({
                    "key": Path(input_path).stem,
                    "text": segment.text
                })

            if not isinstance(asr_result, list) or len(asr_result) == 0:
                return None

        except Exception as e:
            raise RuntimeError(f"语音识别&标点恢复失败: {str(e)}") from e

        return asr_result, info.language

class ASRFactory:
    # 定义语言码与ASR类的映射关系
    ASR_MAP = {
        'zh': ChineseASR,
        'can': CantoneseASR,  # 粤语
        'en': FasterWhisperASR,
        'ja': FasterWhisperASR,
        'ko': FasterWhisperASR
    }

    @classmethod
    def get_asr(cls, language_code, *args, **kwargs):
        """根据语言码返回asr实例"""
        asr_class = cls.ASR_MAP.get(language_code)
        if not asr_class:
            raise ValueError(f"Unsupported language code: {language_code}")
        return asr_class(language=language_code, *args, **kwargs)

class AutoSpeechRecognition:
    '''
    每个语言的ASR类, 都需要实现 execute 方法
    '''
    def __init__(self, language, *args, **kwargs):
        self.asr = ASRFactory.get_asr(language, *args, **kwargs)

    def execute(self, input_path):
        """
        返回值: 
            asr_result - list [{key, text}]
            language   - str     
        """
        return self.asr.execute(input_path)
    
def load_asr_data(asr_dir: Union[str,Path]) -> Dict:
    """
    解析ASR识别结果文件
    返回格式: [{"key": "文件名", "text": "识别文本"}, ...]
    """
    result = {}
    asr_file = Path(asr_dir) / 'output.json'
    try:
        with open(asr_file, 'r', encoding='utf-8') as f:
            result = json.load(f)
    except (SyntaxError, ValueError) as e:
        BotLogger.error(f"ASR文件格式错误: {str(e)}")
    except FileNotFoundError:
        BotLogger.error(f"ASR文件不存在: {asr_file}")
    return result

def batch_asr(language, file_list: List[str], output_dir: str):
    """批量识别函数
    
    Args:
        language: 语言
        file_list: 降噪文件名数组
        output_dir: 语音识别输出目录
        
    Returns:
        语音识别结果(数组)
        
    Raises:
        RuntimeError: 识别处理失败
    """
    try:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_file = Path(output_dir) / "output.json"
        if output_file.exists(): 
            BotLogger.info(
                f"语音识别已处理: {output_file}"
            )
            return

        asr = AutoSpeechRecognition(language)

        combined_results = []
        for file in file_list:
            results, language = asr.execute(input_path=file)
            for result in results:
                combined_results.extend({
                    "key": result.key,
                    "text": result.text,
                    "language": language
                }) 

        output_data = {
            "version": "v4",
            "results": combined_results            
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
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
    parser.add_argument('file', type=str, help='processed file name')
    parser.add_argument('language', type=str, help='language code')
    parser.set_defaults(language='zh')
    parser.add_argument('--no-denoised', action='store_false', dest='denoised',  # 添加反向参数
                        help='disable denoised processing (default: enable denoised)')
    parser.set_defaults(denoised=True)

    args = parser.parse_args()

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

    asr = AutoSpeechRecognition(args.language)
    combined_results = []
    for file in file_list:
        results, language = asr.execute(input_path=file)
        for result in results:
            combined_results.extend({
                "key": result.key,
                "text": result.text,
                "language": language
            })
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(combined_results, f, ensure_ascii=False, indent=2)
