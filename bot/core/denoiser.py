import torch
import torchaudio
from pathlib import Path
from transformers import AutoModel
from typing import Optional, Tuple
import os
from bot.config import PRETRAINED_DIR, DATA_DIR

class AudioDenoiser:
    def __init__(self,
                 model_name: str = "alextomcat/speech_frcrn_ans_cirm_16k",
                 device: Optional[str] = None,
                 local_files_only: bool = True): 
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            local_files_only=local_files_only,
            cache_dir=os.path.join(PRETRAINED_DIR, model_name)  # 本地模型路径
        ).to(self.device)

        # 音频处理参数
        self.target_sr = 16000
        self.n_fft = 512
        self.hop_length = self.n_fft // 4

    def load_audio(self, file_path: str) -> Tuple[torch.Tensor, int]:
        try:
            waveform, orig_sr = torchaudio.load(file_path)
            if orig_sr != self.target_sr:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=orig_sr,
                    new_freq=self.target_sr
                )
                waveform = resampler(waveform)
            return waveform.to(self.device), self.target_sr
        except Exception as e:
            raise RuntimeError(f"音频加载失败: {str(e)}")

    def _stft(self, waveform: torch.Tensor) -> torch.Tensor:
        """STFT转换"""
        return torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=torch.hann_window(self.n_fft).to(self.device),
            return_complex=True
        )

    def _istft(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """逆STFT转换"""
        return torch.istft(
            spectrogram,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=torch.hann_window(self.n_fft).to(self.device)
        )

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
        
        # 1. 加载音频
        waveform, _ = self.load_audio(input_path)
        
        # 2. 转换为频谱图
        stft = self._stft(waveform)
        mag = torch.abs(stft)
        phase = torch.angle(stft)  
        
        # 3. 转换为模型输入格式
        input_mag = mag.unsqueeze(0).to(self.device)  
        
        # 4. 执行降噪
        with torch.no_grad():
            denoised_mag = self.model(input_mag).squeeze()
        
        # 5. 重建波形
        denoised_stft = denoised_mag * torch.exp(1j * phase.to(self.device))
        denoised_waveform = self._istft(denoised_stft)
        if denoised_waveform.max() > 1.0 or denoised_waveform.min() < -1.0:
            # 归一化
            denoised_waveform = np.clip(denoised_waveform, -1.0, 1.0)
        
        # 6. 保存结果
        output_file = output_path / f"{Path(input_path).name}"
        self._save_audio(denoised_waveform, output_file)
        
        return str(output_file)

    def _save_audio(self, waveform: torch.Tensor, output_path: Path):
        waveform = waveform.cpu().clamp(-1.0, 1.0)  # 确保数值范围正确
        torchaudio.save(
            output_path,
            waveform.unsqueeze(0),  # 添加通道维度
            sample_rate=self.target_sr,
            bits_per_sample=32
        )