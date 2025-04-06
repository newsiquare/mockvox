import torch
from pathlib import Path
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
        self.ans(input_path, output_path=output_path / output_file, device=self.device)
        
        return str(output_file)
    
if __name__ == '__main__':
    import os
    from bot.config import PRETRAINED_DIR
    denoise_model = os.path.join(PRETRAINED_DIR, 'damo/speech_frcrn_ans_cirm_16k')
    denoise_model = denoise_model if os.path.exists(denoise_model) else 'damo/speech_frcrn_ans_cirm_16k'
    denoiser = AudioDenoiser(denoise_model)

    file = '/home/drz/mycode/bot/data/sliced/20250406091646/0000610880_0001048320.wav'
    output_dir = '/home/drz/mycode/bot/data/denoised/20250406091646'
    denoised_file = denoiser.denoise(file, output_dir=output_dir)