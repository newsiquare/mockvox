import torch
from pathlib import Path
import os
from typing import Optional
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from bot.config import DENOISED_ROOT_PATH

class AudioDenoiser:
    def __init__(self,
                 model_name: str = 'damo/speech_frcrn_ans_cirm_16k',
                 device: Optional[str] = None): 
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.ans = pipeline(
            task=Tasks.acoustic_noise_suppression,
            model=model_name)

    def denoise(self, 
            input_path: str,
            output_dir: str = DENOISED_ROOT_PATH) -> str:
        """
        降噪处理
        :param input_path: 输入音频路径
        :param output_dir: 输出目录
        :return: 处理后的文件路径
        """
        if not Path(input_path).exists():
            raise FileNotFoundError(f"输入文件不存在: {input_path}")

        # 创建输出目录
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        output_file = os.path.join(output_path, Path(input_path).name)
                
        # 执行降噪
        self.ans(input_path, output_path=output_path / output_file, device=self.device)
        
        return str(output_file)
