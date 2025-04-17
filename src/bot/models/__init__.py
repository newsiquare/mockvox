# from .SpeechSeparation import BSRoformer, MelBandRoformer
from .HuBERT import *
from .tools import *
from .BaseModules import *
from .core_vq import ResidualVectorQuantization
from .quantize import *
from .attentions import *
from .mrte import MRTE, MELEncoder, SpeakerEncoder
from .SynthesizerTrn import SynthesizerTrn
from .MultiPeriodDiscriminator import MultiPeriodDiscriminator
from .mel import spectrogram_torch, spec_to_mel_torch, mel_spectrogram_torch
from .dataset import TextAudioSpeakerLoader, TextAudioSpeakerCollate, DistributedBucketSampler, BucketSampler
from .loss import discriminator_loss, generator_loss, feature_loss, kl_loss

__all__ = [
    # "BSRoformer", 
    # "MelBandRoformer",
    "CNHubert",
    "SynthesizerTrn",
    "MultiPeriodDiscriminator",

    # loss
    "discriminator_loss",
    "generator_loss",
    "feature_loss",
    "kl_loss",

    # 数据集
    "TextAudioSpeakerLoader",
    "TextAudioSpeakerCollate",
    "DistributedBucketSampler",
    "BucketSampler",

    # mel频谱
    "spectrogram_torch",
    "spec_to_mel_torch",
    "mel_spectrogram_torch",

    # quantize
    "ResidualVectorQuantization",
    "ResidualVectorQuantizer",

    # 工具集函数
    "init_weights",
    "get_padding",
    "intersperse",
    "kl_divergence",
    "rand_gumbel",
    "rand_gumbel_like",
    "slice_segments",
    "rand_slice_segments",
    "get_timing_signal_1d",
    "add_timing_signal_1d",
    "cat_timing_signal_1d",
    "subsequent_mask",
    "fused_add_tanh_sigmoid_multiply",
    "convert_pad_shape",
    "shift_1d",
    "sequence_mask",
    "generate_path",
    "clip_grad_value_",
    "squeeze",
    "unsqueeze",

    # 标准化流模型（Normalizing Flows）的关键组件
    "rational_quadratic_spline",
    "unconstrained_rational_quadratic_spline",
    "searchsorted",
    "piecewise_rational_quadratic_transform",

    # Base Modules
    "LayerNorm",
    "ConvReluNorm",
    "DDSConv",
    "WaveNet",    
    "ResBlock1",
    "ResBlock2",    
    "MultiHeadAttention",
    "ScaledDotProductAttention",    
    "MelStyleEncoder",
    "MelStyleEncoderVAE",    
    "ConvFlow",
    "ResidualCouplingLayer",
    "ActNorm",
    "InvConvNear",
    "Log",
    "Flip",
    "ElementwiseAffine",    
    "Conv1dGLU",
    "ConvNorm",
    "LinearNorm",
    "Mish",

    # 注意力
    "Encoder",
    "Decoder"
    "MultiHeadAttention"
    "FFN",
    "FFT",
    "Depthwise_Separable_Conv1D",
    "Depthwise_Separable_TransposeConv1D",
    "TransformerCouplingLayer",

    # MRTE
    "MRTE",
    "MELEncoder",
    "SpeakerEncoder"
]