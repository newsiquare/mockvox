import torch
from pathlib import Path
import os
import gc
from typing import Optional, List
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

from bot.config import DENOISED_ROOT_PATH, PRETRAINED_PATH
from bot.utils import BotLogger

class AudioDenoiser:
    def __init__(self,
                 model_name: str = 'damo/speech_frcrn_ans_cirm_16k',
                 device: Optional[str] = None): 
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.ans = pipeline(
            task=Tasks.acoustic_noise_suppression,
            model=os.path.join(PRETRAINED_PATH,model_name)
        )

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
        
        del denoiser
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.ipc_collect()
            gc.collect()
        return denoised_files
    
    except Exception as e:
        BotLogger.error(
            f"Denoise failed: {output_dir} \nException: {str(e)}",
            extra={"action": "denoise_error"}
        )
        raise RuntimeError(f"Denoise failed: {str(e)}") from e
