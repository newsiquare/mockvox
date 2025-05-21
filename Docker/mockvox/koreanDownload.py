from modelscope import snapshot_download

model_dir = snapshot_download("iic/speech_UniASR_asr_2pass-ko-16k-common-vocab6400-tensorflow1-offline",local_dir="/mockvox/pretrained/iic/speech_UniASR_asr_2pass-ko-16k-common-vocab6400-tensorflow1-offline")