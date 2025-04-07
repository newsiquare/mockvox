from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from bot.config import PRETRAINED_DIR, DENOISED_ROOT_PATH
from typing import Optional, List
import torch
import os

class AutoSpeechRecognition:
    def __init__(self,
                 model_name: str = 'iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch',
                 device: Optional[str] = None): 
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.asr = pipeline(
            task=Tasks.auto_speech_recognition,
            model=model_name,
            model_revision="v2.0.4",
            preprocessor=None,
            device=self.device)
        
    def speech_recognition(self, 
            input_path: str) -> List:
        result = self.asr(input=input_path)
        return result
