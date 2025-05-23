import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
import librosa
from librosa.filters import mel as librosa_mel_fn
from typing import Optional

from bot.utils import BotLogger

MAX_WAV_VALUE = 32768.0

# -------------------
# 动态范围压缩/解压
# -------------------
def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    """对数动态范围压缩
    Args:
        x: 输入张量
        C: 压缩因子 (默认1)
        clip_val: 避免log(0)的最小裁剪值
    Returns:
        压缩后的对数幅度
    """
    return torch.log(torch.clamp(x, min=clip_val) * C)

def dynamic_range_decompression_torch(x, C=1):
    """对数动态范围解压缩
    Args:
        x: 压缩后的张量
        C: 压缩时使用的因子
    """
    return torch.exp(x) / C

# -------------------
# 频谱归一化相关
# -------------------
def spectral_normalize_torch(magnitudes):
    """频谱幅度归一化（封装压缩函数）"""
    return dynamic_range_compression_torch(magnitudes)

def spectral_de_normalize_torch(magnitudes):
    """频谱幅度反归一化（封装解压函数）"""
    return dynamic_range_decompression_torch(magnitudes)

# -------------------
# 全局缓存字典（用于存储预计算的滤波器组和窗函数）
# -------------------
mel_basis = {}  # 缓存Mel滤波器组
hann_window = {}  # 缓存汉宁窗

# -------------------
# 频谱计算函数
# -------------------
def spectrogram_torch(y, n_fft, hop_size, win_size, center: Optional[bool] = False):
    """计算幅度谱
    Args:
        y: 音频波形 (Tensor) shape: (B, T)
        n_fft: FFT点数
        sampling_rate: 采样率（未实际使用，可考虑移除）
        hop_size: 帧移
        win_size: 窗长
        center: 是否中心填充
    Returns:
        spec: 幅度谱 shape: (B, n_fft//2+1, T)
    """
    # 输入范围检查
    if (y.abs().max() > 1.0 + 1e-3):  # 改为 1.0 * (1 + 1e-3)
        BotLogger.warn(f"Input amplitude abnormal: max={y.abs().max():.3f}")

    # 获取设备相关缓存键
    global hann_window
    dtype_device = str(y.dtype) + "_" + str(y.device)
    wnsize_dtype_device = str(win_size) + "_" + dtype_device
    
    # 缓存汉宁窗
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(
            dtype=y.dtype, device=y.device
        )

    # 反射填充防止边界效应
    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)

    # 计算STFT（返回复数谱）
    spec_complex = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window[wnsize_dtype_device],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    return torch.abs(spec_complex)

def spec_to_mel_torch(spec, n_fft, num_mels, sampling_rate, fmin, fmax):
    """将线性频谱转换为Mel频谱
    Args:
        spec: 输入幅度谱, shape=(B, n_freq, T)
        n_fft: FFT点数
        num_mels: Mel滤波器数量
        sampling_rate: 采样率
        fmin: 最低Mel频率
        fmax: 最高Mel频率
    Returns:
        mel_spec: Mel频谱, shape=(B, num_mels, T)
    """
    global mel_basis
    dtype_device = str(spec.dtype) + "_" + str(spec.device)
    
    # 唯一缓存键
    cache_key = f"{fmin}_{fmax}_{num_mels}_{n_fft}_{dtype_device}"
    
    if cache_key not in mel_basis:
        # 生成Mel滤波器矩阵
        fmax = min(fmax, sampling_rate//2)
        mel = librosa_mel_fn(
            sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax
        )
        mel_basis[cache_key] = torch.from_numpy(mel).to(
            dtype=spec.dtype, device=spec.device
        )
    
    # 执行矩阵乘法：mel_fb (n_mels, n_freq) @ spec (B, n_freq, T) -> (B, n_mels, T)
    mel_spec = torch.matmul(mel_basis[cache_key], spec)
    
    # 应用谱归一化（与librosa的log压缩对齐）
    return spectral_normalize_torch(mel_spec)

def mel_spectrogram_torch(
        y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center: Optional[bool] = False
    ):
    # 输入检查
    if (y.abs().max() > 1.0 + 1e-3):
        BotLogger.warn(f"Input amplitude abnormal: max={y.abs().max():.3f}")

    spec = spectrogram_torch(y, n_fft, hop_size, win_size, center)
    spec = spec_to_mel_torch(spec, n_fft, num_mels, sampling_rate, fmin, fmax)
    return spec