# -*- coding: utf-8 -*-
import ast
import os
from typing import Optional, List
from pathlib import Path
import torch
import torchaudio
from bot.config import ASR_PATH, PROCESS_PATH, DENOISED_ROOT_PATH, SLICED_ROOT_PATH
from bot.core import load_audio
import numpy as np
from bot.utils import BotLogger
from bot.models import CNHubert
from scipy.io import wavfile

class FeatureExtractor:
    def __init__(self,
                 maxx=0.95,
                 alpha=0.5,
                 device: Optional[str] = None
        ):    
        self.resampler = torchaudio.transforms.Resample(
            orig_freq=32000,
            new_freq=16000,
            resampling_method='sinc_interpolation',
            lowpass_filter_width=16,
            rolloff=0.85,
            beta=5.0
        )
        self.maxx = maxx
        self.alpha = alpha
        self.nan_fails = []

        # 设备配置（优先使用GPU）
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CNHubert().to(self.device)

    @staticmethod
    def load_asr_data(asr_file):
        """读取ASR结果文件并还原为列表"""
        result = []
        with open(asr_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue  # 跳过空行
                try:
                    # 使用 ast.literal_eval 安全转换字符串为字典
                    result.append(ast.literal_eval(line))
                except SyntaxError as e:
                    print(f"格式错误的行: {line}\n错误信息: {e}")
        return result
    
    def extract(self, file_path, denoised=True) -> List:
        # 路径配置
        asr_dir = os.path.join(ASR_PATH, file_path)
        if(denoised):
            wav_dir = os.path.join(DENOISED_ROOT_PATH, file_path)
        else:
            wav_dir = os.path.join(SLICED_ROOT_PATH, file_path)

        processed_dir = os.path.join(PROCESS_PATH, file_path)
        hubert_dir = os.path.join(processed_dir, "cnhubert")
        Path(hubert_dir).mkdir(parents=True, exist_ok=True)
        wav32_dir = os.path.join(processed_dir, "wav32k")
        Path(wav32_dir).mkdir(parents=True, exist_ok=True)

        # 加载ASR数据
        asr_file = os.path.join(asr_dir, 'output.txt')
        lines = self.load_asr_data(asr_file)

        for line in lines:
            wav_file = f"{wav_dir}/{line['key']}.wav"
            ssl = self.name2go(wav_file, wav32_dir, hubert_dir)

    def name2go(self, wav_file_path, wav32k_dir, cnhubert_dir):
        tmp_audio = load_audio(wav_file_path, 32000)
        tmp_max = np.abs(tmp_audio).max()
        if tmp_max > 2.2:
            BotLogger.info(f"{wav_file_path}-filtered, max audio {tmp_max} extend 2.2")
            return
        
        tmp_audio32 = (tmp_audio / tmp_max * (self.maxx * self.alpha*32768)) \
            + ((1 - self.alpha)*32768) * tmp_audio
        tmp_audio32b = (tmp_audio / tmp_max * (self.maxx * self.alpha*1145.14)) \
            + ((1 - self.alpha)*1145.14) * tmp_audio
        
        audio_tensor = torch.from_numpy(tmp_audio32b).reshape(1, -1)
        tensor_wav16 = self.resampler(audio_tensor).to(self.device)

        ssl = self.model.model(tensor_wav16)["last_hidden_state"].transpose(1,2).cpu()

        if np.isnan(ssl.detach().numpy()).sum() != 0:
            # self.nan_fails.append(wav_file_path)
            BotLogger.info(f"nan filtered: {wav_file_path}")
            return

        wavfile.write(
            f"{wav32k_dir}/{Path(wav_file_path).name}",
            32000,
            tmp_audio32.astype("int16"),
        )        
        torch.save(ssl, f"{cnhubert_dir}/{Path(wav_file_path).stem}.pth")

if __name__ == '__main__':
    extractor = FeatureExtractor()
    extractor.extract('20250409145258452558.1ed301dd.788fc313bf38482aa63fe2ea09781878', denoised=True)
