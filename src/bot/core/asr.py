# -*- coding: utf-8 -*-
"""
语音识别(Auto Speech Recognition, asr)模块
"""
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from typing import Optional, List, Union
from pathlib import Path
import torch
import json
from bot.utils import BotLogger

class AutoSpeechRecognition:
    def __init__(self,
                 asr_model_name: str = 'iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch',
                 punc_model_name: str = 'iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch',
                 device: Optional[str] = None
        ): 
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # 语音识别管道
        self.asr = pipeline(
            task=Tasks.auto_speech_recognition,
            model=asr_model_name,
            model_revision="v2.0.4",
            preprocessor=None,
            device=self.device,
            disable_update=True
        )
        
        # 标点恢复管道
        self.punc = pipeline(
            task=Tasks.punctuation,
            model=punc_model_name,
            model_revision="v2.0.4",
            preprocessor=None,
            device=self.device,
            sequence_length=512,
            batch_size=8,
            disable_update=True
        )
        
    def speech_recognition(self, input_path: str) -> List:
        try:
            asr_result = self.asr(input=input_path, nbest=1, beam_size=10)
            if not isinstance(asr_result, list) or len(asr_result) == 0:
                raise ValueError("ASR结果必须是包含至少一个元素的列表")

            texts = [item['text'] for item in asr_result]
            batch_punc = self.punc(texts, batch_size=8)

            if len(batch_punc) != len(asr_result):
                raise RuntimeError(
                    f"标点恢复输入输出数量不匹配，输入{len(asr_result)}条，输出{len(batch_punc)}条"
                )

            for item, punc_item in zip(asr_result, batch_punc):
                item['text'] = punc_item['text']
        
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
