from modelscope import snapshot_download

model_dir = snapshot_download("iic/speech_UniASR_asr_2pass-cantonese-CHS-16k-common-vocab1468-tensorflow1-online",local_dir="/mockvox/pretrained/iic/speech_UniASR_asr_2pass-cantonese-CHS-16k-common-vocab1468-tensorflow1-online")