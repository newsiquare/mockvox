# -*- coding: utf-8 -*-
"""
工具集
"""
import math
import torch
from torch.nn import functional as F
import numpy as np

def init_weights(m, mean=0.0, std=0.01):
    """初始化卷积层权重（正态分布）
    Args:
        m: 待初始化的模块
        mean: 正态分布均值
        std: 正态分布标准差
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)

def get_padding(kernel_size, dilation=1):
    """计算保持时序长度不变的填充大小
    公式推导: padding = (k*d - d)/2 = d*(k-1)/2
    """
    return int((kernel_size * dilation - dilation) / 2)

def intersperse(lst, item):
    """在列表元素之间插入指定项
    示例: intersperse([1,2,3], 0) => [0,1,0,2,0,3,0]
    """
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result

def kl_divergence(m_p, logs_p, m_q, logs_q):
    """计算两个高斯分布之间的KL散度(KL(P||Q))
    """
    kl = (logs_q - logs_p) - 0.5
    kl += (
        0.5 * (torch.exp(2.0 * logs_p) + ((m_p - m_q) ** 2)) * torch.exp(-2.0 * logs_q)
    )
    return kl

def rand_gumbel(shape):
    """从Gumbel分布采样(用于Gumbel-Softmax)
    保护机制：限制采样范围[0.00001, 0.99999]防止log(0)
    """
    uniform_samples = torch.rand(shape) * 0.99998 + 0.00001
    return -torch.log(-torch.log(uniform_samples))

def rand_gumbel_like(x):
    """生成与输入张量相同设备/精度的Gumbel样本"""
    g = rand_gumbel(x.size()).to(dtype=x.dtype, device=x.device)
    return g

def slice_segments(x, ids_str, segment_size=4):
    """从张量中切片固定长度的片段
    Args:
        x: 输入张量 (B, D, T)
        ids_str: 各样本的起始索引 (B,)
        segment_size: 切片长度
    """
    ret = torch.zeros_like(x[:, :, :segment_size])
    for i in range(x.size(0)):
        idx_str = ids_str[i]
        idx_end = idx_str + segment_size
        ret[i] = x[i, :, idx_str:idx_end]
    return ret

def rand_slice_segments(x, x_lengths=None, segment_size=4):
    """随机切片片段（支持变长输入）
    Returns:
        ret: 切片结果 (B, D, segment_size)
        ids_str: 实际使用的起始索引
    """
    b, d, t = x.size()
    if x_lengths is None:
        x_lengths = t
    ids_str_max = x_lengths - segment_size + 1
    ids_str = (torch.rand([b]).to(device=x.device) * ids_str_max).to(dtype=torch.long)
    ret = slice_segments(x, ids_str, segment_size)
    return ret, ids_str

def get_timing_signal_1d(length, channels, min_timescale=1.0, max_timescale=1.0e4):
    """生成1D位置编码(类似Transformer位置嵌入)
    Args:
        length: 序列长度
        channels: 编码维度
        min/max_timescale: 控制频率范围
    """
    position = torch.arange(length, dtype=torch.float)
    num_timescales = channels // 2
    log_timescale_increment = math.log(max_timescale / min_timescale) / (num_timescales - 1)
    inv_timescales = min_timescale * torch.exp(torch.arange(num_timescales) * -log_timescale_increment)
    
    scaled_time = position.unsqueeze(0) * inv_timescales.unsqueeze(1)  # (num_timescales, length)
    signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], 0)  # (2*num_timescales, length)
    signal = F.pad(signal, [0, 0, 0, channels % 2])  # 补零到目标维度
    return signal.view(1, channels, length)

def add_timing_signal_1d(x, min_timescale=1.0, max_timescale=1.0e4):
    """将位置编码添加到输入(加法融合)"""
    b, channels, length = x.size()
    signal = get_timing_signal_1d(length, channels, min_timescale, max_timescale)
    return x + signal.to(dtype=x.dtype, device=x.device)

def cat_timing_signal_1d(x, min_timescale=1.0, max_timescale=1.0e4, axis=1):
    """拼接位置编码到输入(沿指定轴)"""
    b, channels, length = x.size()
    signal = get_timing_signal_1d(length, channels, min_timescale, max_timescale)
    return torch.cat([x, signal.to(dtype=x.dtype, device=x.device)], axis)

def subsequent_mask(length):
    """生成因果掩码(下三角矩阵)
    用于自回归模型防止看到未来信息
    """
    mask = torch.tril(torch.ones(length, length)).unsqueeze(0).unsqueeze(0)
    return mask

@torch.jit.script
def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
    """融合操作: tanh(a+b) * sigmoid(a+b)
    优化技巧: 使用TorchScript加速计算
    """
    n_channels_int = n_channels[0]
    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :n_channels_int, :])
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
    return t_act * s_act

def convert_pad_shape(pad_shape):
    """转换填充形状为PyTorch格式
    示例：[[0,0], [1,1], [2,2]] → [2,2,1,1,0,0]
    """
    return [item for sublist in pad_shape[::-1] for item in sublist]

def shift_1d(x):
    """时序右移一位（用于因果卷积）"""
    x = F.pad(x, convert_pad_shape([[0, 0], [0, 0], [1, 0]]))[:, :, :-1]
    return x

def sequence_mask(length, max_length=None):
    """生成序列掩码（标记有效位置）
    Args:
        length: 各样本实际长度 (B,)
    Returns: (B, max_length) 的布尔掩码
    """
    if max_length is None:
        max_length = length.max()
    return torch.arange(max_length, device=length.device)[None, :] < length[:, None]

def generate_path(duration, mask):
    """根据持续时间生成对齐路径
    Args:
        duration: 持续时间 (B, 1, T_x)
        mask: 原始掩码 (B, 1, T_y, T_x)
    Returns: 对齐路径矩阵 (B, 1, T_y, T_x)
    """
    b, _, t_y, t_x = mask.shape
    cum_duration = torch.cumsum(duration, -1)  # 累积持续时间
    
    # 生成平坦化的序列掩码
    path = sequence_mask(cum_duration.view(b * t_x), t_y).view(b, t_x, t_y)
    
    # 计算路径变化点
    path = path - F.pad(path, convert_pad_shape([[0, 0], [1, 0], [0, 0]]))[:, :-1]
    return path.unsqueeze(1).transpose(2, 3) * mask

def clip_grad_value_(parameters, clip_value, norm_type=2):
    """梯度裁剪 (按值裁剪+梯度范数计算)
    Args:
        clip_value: 裁剪阈值(None表示不裁剪)
        norm_type: 范数类型(2=L2)
    Returns: 总梯度范数
    """
    parameters = [p for p in parameters if p.grad is not None]
    total_norm = 0
    
    for p in parameters:
        if clip_value is not None:
            p.grad.data.clamp_(min=-clip_value, max=clip_value)
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
        
    return total_norm ** (1.0 / norm_type)

def squeeze(x, x_mask=None, n_sqz=2):
    """压缩时序维度 (合并相邻帧)
    Args:
        x: (B, C, T)
        n_sqz: 压缩倍数
    Returns: (B, C*n_sqz, T//n_sqz)
    """
    b, c, t = x.size()
    t = (t // n_sqz) * n_sqz  # 对齐长度
    x_sqz = x[:,:,:t].view(b, c, t//n_sqz, n_sqz).permute(0,3,1,2).reshape(b, c*n_sqz, t//n_sqz)
    
    if x_mask is not None:
        x_mask = x_mask[:, :, n_sqz-1::n_sqz]  # 保留最后时刻的掩码
    return x_sqz, x_mask if x_mask else torch.ones(b,1,t//n_sqz, device=x.device)

def unsqueeze(x, x_mask=None, n_sqz=2):
    """解压缩时序维度 (逆squeeze操作)"""
    b, c, t = x.size()
    x_unsqz = x.view(b, n_sqz, c//n_sqz, t).permute(0,2,3,1).reshape(b, c//n_sqz, t*n_sqz)
    
    if x_mask is not None:
        x_mask = x_mask.unsqueeze(-1).repeat(1,1,1,n_sqz).view(b,1,t*n_sqz)
    return x_unsqz, x_mask if x_mask else torch.ones(b,1,t*n_sqz, device=x.device)

import torch
from torch.nn import functional as F
import numpy as np

# 默认最小参数设置（防止数值不稳定）
DEFAULT_MIN_BIN_WIDTH = 1e-3    # 最小分箱宽度
DEFAULT_MIN_BIN_HEIGHT = 1e-3   # 最小分箱高度
DEFAULT_MIN_DERIVATIVE = 1e-3   # 最小导数值

def piecewise_rational_quadratic_transform(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    unnormalized_derivatives,
    inverse=False,
    tails=None,
    tail_bound=1.0,
    min_bin_width=DEFAULT_MIN_BIN_WIDTH,
    min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
    min_derivative=DEFAULT_MIN_DERIVATIVE,
):
    """分段有理二次样条变换（核心变换函数）
    参数：
        inputs: 输入张量
        unnormalized_widths: 未归一化的分箱宽度参数 (将用softmax归一化)
        unnormalized_heights: 未归一化的分箱高度参数
        unnormalized_derivatives: 未归一化的导数值参数 (将用softplus处理)
        inverse: 是否进行逆变换
        tails: 边界处理方式（'linear'表示线性外推）
        tail_bound: 边界阈值
    Returns:
        outputs: 变换后的张量
        logabsdet: 对数绝对雅可比行列式
    """
    # 根据边界设置选择不同的样条实现
    if tails is None:
        spline_fn = rational_quadratic_spline  # 有界区域标准样条
        spline_kwargs = {}
    else:
        spline_fn = unconstrained_rational_quadratic_spline  # 带边界处理的样条
        spline_kwargs = {"tails": tails, "tail_bound": tail_bound}

    # 执行选定的样条变换
    outputs, logabsdet = spline_fn(
        inputs=inputs,
        unnormalized_widths=unnormalized_widths,
        unnormalized_heights=unnormalized_heights,
        unnormalized_derivatives=unnormalized_derivatives,
        inverse=inverse,
        min_bin_width=min_bin_width,
        min_bin_height=min_bin_height,
        min_derivative=min_derivative,
        **spline_kwargs
    )
    return outputs, logabsdet

def searchsorted(bin_locations, inputs, eps=1e-6):
    """分箱位置搜索函数 (类似numpy.searchsorted)
    参数：
        bin_locations: 分箱边界位置
        inputs: 需要查询的输入值
        eps: 防止数值溢出的微小值
    Returns:
        每个输入值对应的分箱索引
    """
    bin_locations[..., -1] += eps  # 避免右边界相等的情况
    return torch.sum(inputs[..., None] >= bin_locations, dim=-1) - 1

def unconstrained_rational_quadratic_spline(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    unnormalized_derivatives,
    inverse=False,
    tails="linear",
    tail_bound=1.0,
    min_bin_width=DEFAULT_MIN_BIN_WIDTH,
    min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
    min_derivative=DEFAULT_MIN_DERIVATIVE,
):
    """无约束有理二次样条（带边界外推处理）
    实现参考: Conor Durkan et al. Neural Spline Flows (2019)
    """
    # 创建掩码区分边界内外区域
    inside_interval_mask = (inputs >= -tail_bound) & (inputs <= tail_bound)
    outside_interval_mask = ~inside_interval_mask

    outputs = torch.zeros_like(inputs)
    logabsdet = torch.zeros_like(inputs)

    if tails == "linear":
        # 为导数添加边界padding
        unnormalized_derivatives = F.pad(unnormalized_derivatives, pad=(1, 1))
        # 边界导数约束（保证线性外推）
        constant = np.log(np.exp(1 - min_derivative) - 1)
        unnormalized_derivatives[..., 0] = constant
        unnormalized_derivatives[..., -1] = constant

        # 边界外区域保持线性变换
        outputs[outside_interval_mask] = inputs[outside_interval_mask]
        logabsdet[outside_interval_mask] = 0  # 雅可比行列式为1，故log为0
    else:
        raise RuntimeError(f"{tails} tails are not implemented.")

    # 内部区域使用标准有理二次样条
    (outputs[inside_interval_mask], logabsdet[inside_interval_mask]) = rational_quadratic_spline(
        inputs=inputs[inside_interval_mask],
        unnormalized_widths=unnormalized_widths[inside_interval_mask, :],
        unnormalized_heights=unnormalized_heights[inside_interval_mask, :],
        unnormalized_derivatives=unnormalized_derivatives[inside_interval_mask, :],
        inverse=inverse,
        left=-tail_bound,
        right=tail_bound,
        bottom=-tail_bound,
        top=tail_bound,
        min_bin_width=min_bin_width,
        min_bin_height=min_bin_height,
        min_derivative=min_derivative,
    )

    return outputs, logabsdet

def rational_quadratic_spline(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    unnormalized_derivatives,
    inverse=False,
    left=0.0,
    right=1.0,
    bottom=0.0,
    top=1.0,
    min_bin_width=DEFAULT_MIN_BIN_WIDTH,
    min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
    min_derivative=DEFAULT_MIN_DERIVATIVE,
):
    """标准有理二次样条变换（处理有界区域）
    核心公式参考: https://arxiv.org/abs/1906.04032
    """
    # 输入范围验证
    if torch.min(inputs) < left or torch.max(inputs) > right:
        raise ValueError("Input out of domain")

    num_bins = unnormalized_widths.shape[-1]
    
    # 参数有效性检查
    if min_bin_width * num_bins > 1.0:
        raise ValueError("Min bin width exceeds total space")
    if min_bin_height * num_bins > 1.0:
        raise ValueError("Min bin height exceeds total space")

    # 归一化分箱参数
    widths = F.softmax(unnormalized_widths, dim=-1)
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths
    cumwidths = torch.cumsum(widths, dim=-1)
    cumwidths = F.pad(cumwidths, pad=(1, 0), value=0.0)
    cumwidths = (right - left) * cumwidths + left  # 缩放至指定区间
    cumwidths[..., [0, -1]] = torch.tensor([left, right])  # 固定边界
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]  # 计算实际宽度

    # 处理导数参数
    derivatives = min_derivative + F.softplus(unnormalized_derivatives)

    # 归一化高度参数
    heights = F.softmax(unnormalized_heights, dim=-1)
    heights = min_bin_height + (1 - min_bin_height * num_bins) * heights
    cumheights = torch.cumsum(heights, dim=-1)
    cumheights = F.pad(cumheights, pad=(1, 0), value=0.0)
    cumheights = (top - bottom) * cumheights + bottom  # 缩放至指定区间
    cumheights[..., [0, -1]] = torch.tensor([bottom, top])  # 固定边界
    heights = cumheights[..., 1:] - cumheights[..., :-1]  # 计算实际高度

    # 确定输入所在的分箱索引
    bin_idx = searchsorted(cumheights if inverse else cumwidths, inputs)[..., None]

    # 提取当前分箱的参数
    input_cumwidths = cumwidths.gather(-1, bin_idx)[..., 0]
    input_bin_widths = widths.gather(-1, bin_idx)[..., 0]
    input_cumheights = cumheights.gather(-1, bin_idx)[..., 0]
    input_delta = (heights / widths).gather(-1, bin_idx)[..., 0]
    input_derivatives = derivatives.gather(-1, bin_idx)[..., 0]
    input_derivatives_plus_one = derivatives[..., 1:].gather(-1, bin_idx)[..., 0]
    input_heights = heights.gather(-1, bin_idx)[..., 0]

    if inverse:
        # 逆变换：求解二次方程
        numerator = (inputs - input_cumheights) * (
            input_derivatives + input_derivatives_plus_one - 2 * input_delta
        )
        denominator = input_heights * (input_delta - input_derivatives) - numerator
        root = 2 * numerator / (-denominator - torch.sqrt(denominant**2 - 4 * numerator * input_delta * (inputs - input_cumheights)))
        
        outputs = root * input_bin_widths + input_cumwidths
        # 计算雅可比行列式
        theta_one_minus_theta = root * (1 - root)
        denominator = input_delta + (input_derivatives + input_derivatives_plus_one - 2 * input_delta) * theta_one_minus_theta
        derivative_numerator = input_delta**2 * (input_derivatives_plus_one * root**2 + 2 * input_delta * theta_one_minus_theta + input_derivatives * (1 - root)**2)
        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)
        return outputs, -logabsdet
    else:
        # 前向变换
        theta = (inputs - input_cumwidths) / input_bin_widths
        theta_one_minus_theta = theta * (1 - theta)
        
        # 有理二次插值
        numerator = input_heights * (input_delta * theta**2 + input_derivatives * theta_one_minus_theta)
        denominator = input_delta + (input_derivatives + input_derivatives_plus_one - 2 * input_delta) * theta_one_minus_theta
        outputs = input_cumheights + numerator / denominator
        
        # 计算雅可比行列式
        derivative_numerator = input_delta**2 * (input_derivatives_plus_one * theta**2 + 2 * input_delta * theta_one_minus_theta + input_derivatives * (1 - theta)**2)
        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)
        return outputs, logabsdet