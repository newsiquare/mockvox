from modelscope import snapshot_download

model_dir = snapshot_download("iic/speech_UniASR_asr_2pass-ja-16k-common-vocab93-tensorflow1-offline",local_dir="/mockvox/pretrained/iic/speech_UniASR_asr_2pass-ja-16k-common-vocab93-tensorflow1-offline")