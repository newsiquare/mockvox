import torch
import torchaudio
from pathlib import Path
from typing import Optional, Tuple
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import os
from bot.config import PRETRAINED_DIR, DATA_DIR

class AudioDenoiser:
    def __init__(self,
                 model_name: str = 'iic/speech_frcrn_ans_cirm_16k',
                 device: Optional[str] = None): 
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.ans = pipeline(
            Tasks.acoustic_noise_suppression,
            model=model_name)       

    def denoise(self, 
            input_path: str,
            output_dir: str = os.path.join(DATA_DIR, "denoised")) -> str:
        """
        全流程降噪处理 (torchaudio版)
        :param input_path: 输入音频路径
        :param output_dir: 输出目录
        :return: 处理后的文件路径
        """
        if not Path(input_path).exists():
            raise FileNotFoundError(f"输入文件不存在: {input_path}")

        # 创建输出目录
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        output_file = Path(input_path).name
                
        # 执行降噪
        result = self.ans(input_path, output_path / output_file, device=self.device)
        
        return str(output_file)