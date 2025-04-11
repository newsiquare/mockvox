# -*- coding: utf-8 -*-
"""音频特征提取模块，基于CNHubert模型实现语音特征提取"""

import ast
import os
from typing import Optional, List
from pathlib import Path
import torch
import torchaudio
import numpy as np
from scipy.io import wavfile

from bot.config import ASR_PATH, PROCESS_PATH, DENOISED_ROOT_PATH, SLICED_ROOT_PATH
from bot.core import load_audio
from bot.utils import BotLogger
from bot.models import CNHubert

class FeatureExtractor:
    """音频特征提取器，负责处理音频文件并提取Hubert特征"""
    
    def __init__(self,
                 maxx: float = 0.95,  # 音频归一化最大系数
                 alpha: float = 0.5,  # 新旧音频增益混合比例
                 device: Optional[str] = None  # 指定计算设备
        ):
        # 初始化重采样器（32kHz -> 16kHz）
        self.resampler = torchaudio.transforms.Resample(
            orig_freq=32000,
            new_freq=16000,
            resampling_method='sinc_interpolation',
            lowpass_filter_width=16,
            rolloff=0.85,
            beta=5.0  # Kaiser窗口参数
        )
        
        # 音频处理参数
        self.maxx = maxx
        self.alpha = alpha
        self.nan_fails = []  # 记录处理失败的文件

        # 设备配置（自动检测GPU）
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # 加载预训练模型到指定设备
        self.model = CNHubert().to(self.device).eval()  # 添加eval模式

    @staticmethod
    def load_asr_data(asr_file: str) -> List[dict]:
        """
        解析ASR识别结果文件
        返回格式: [{"key": "文件名", "text": "识别文本"}, ...]
        """
        result = []
        try:
            with open(asr_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    cleaned_line = line.strip()
                    if not cleaned_line:
                        continue
                    try:
                        result.append(ast.literal_eval(cleaned_line))
                    except (SyntaxError, ValueError) as e:
                        BotLogger.error(f"ASR文件格式错误 行号:{line_num} 内容:{cleaned_line} 错误:{str(e)}")
        except FileNotFoundError:
            BotLogger.error(f"ASR文件不存在: {asr_file}")
        return result
    
    def extract(self, file_path: str, denoised: bool = True) -> List:
        """
        主处理流程
        :param file_path: 相对路径标识符
        :param denoised: 是否使用降噪后的音频
        :return: 处理结果列表（当前占位）
        """
        # 构建目录路径
        asr_dir = Path(ASR_PATH) / file_path
        wav_root = DENOISED_ROOT_PATH if denoised else SLICED_ROOT_PATH
        wav_dir = Path(wav_root) / file_path

        # 创建输出目录
        processed_dir = Path(PROCESS_PATH) / file_path
        hubert_dir = processed_dir / "cnhubert"
        wav32_dir = processed_dir / "wav32k"
        hubert_dir.mkdir(parents=True, exist_ok=True)
        wav32_dir.mkdir(parents=True, exist_ok=True)

        # 处理ASR结果
        asr_file = asr_dir / 'output.txt'
        for line in self.load_asr_data(str(asr_file)):
            wav_file = wav_dir / f"{line['key']}.wav"
            if not wav_file.exists():
                BotLogger.warning(f"音频文件不存在: {wav_file}")
                continue
                
            # 执行特征提取
            self._process_audio(
                wav_file_path=str(wav_file),
                wav32k_dir=str(wav32_dir),
                cnhubert_dir=str(hubert_dir)
            )

    def _process_audio(self, wav_file_path: str, wav32k_dir: str, cnhubert_dir: str):
        """
        核心音频处理流程
        :param wav_file_path: 输入音频路径
        :param wav32k_dir: 32kHz音频输出目录
        :param cnhubert_dir: 特征文件输出目录
        """
        try:
            # 加载并校验音频
            audio, sr = load_audio(wav_file_path, 32000)
            if audio is None:
                BotLogger.error(f"音频加载失败: {wav_file_path}")
                return

            # 音频幅值校验
            max_amplitude = np.abs(audio).max()
            if max_amplitude > 2.2:
                BotLogger.info(f"幅值过大被过滤: {wav_file_path} (峰值: {max_amplitude:.2f})")
                return

            # 音频增益混合处理
            scaled_audio = self._audio_scaling(audio, max_amplitude)
            
            # 生成16kHz重采样音频（设备自动处理）
            audio_tensor = torch.from_numpy(scaled_audio).reshape(1, -1)
            tensor_wav16 = self.resampler(audio_tensor).to(self.device)

            # 特征提取（batch_size=1）
            with torch.no_grad():  # 禁用梯度计算
                hidden_states = self.model(tensor_wav16)["last_hidden_state"]
                ssl = hidden_states.transpose(1, 2).cpu()

            # 特征校验
            if torch.isnan(ssl).any():
                BotLogger.info(f"NaN特征被过滤: {wav_file_path}")
                return

            # 输出处理结果
            self._save_outputs(
                wav32k_dir=wav32k_dir,
                cnhubert_dir=cnhubert_dir,
                wav_file_path=wav_file_path,
                scaled_audio=scaled_audio,
                ssl=ssl
            )
            
        except Exception as e:
            BotLogger.error(f"处理异常 {wav_file_path}: {str(e)}")

    def _audio_scaling(self, audio: np.ndarray, max_amp: float) -> np.ndarray:
        """音频增益混合计算"""
        # 混合新旧两种增益策略
        return (audio / max_amp * (self.maxx * self.alpha * 1145.14)) \
                + ((1 - self.alpha) * 1145.14) * audio

    def _save_outputs(self, wav32k_dir: str, cnhubert_dir: str, 
                     wav_file_path: str, scaled_audio: np.ndarray, ssl: torch.Tensor):
        """保存处理结果"""
        # 保存32kHz格式音频
        wav_path = Path(wav32k_dir) / Path(wav_file_path).name
        wavfile.write(
            str(wav_path),
            32000,
            scaled_audio.astype("int16"),
        )
        
        # 保存特征文件
        feature_path = Path(cnhubert_dir) / f"{Path(wav_file_path).stem}.pth"
        torch.save(ssl, str(feature_path))
        BotLogger.debug(f"处理完成: {wav_file_path} -> {feature_path}")

if __name__ == '__main__':
    # 示例用法
    extractor = FeatureExtractor()
    extractor.extract('20250409145258452558.1ed301dd.788fc313bf38482aa63fe2ea09781878', denoised=True)