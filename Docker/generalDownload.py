from modelscope import snapshot_download

model_dir = snapshot_download("damo/speech_frcrn_ans_cirm_16k",local_dir="/mockvox/pretrained/damo/speech_frcrn_ans_cirm_16k")
model_dir = snapshot_download("iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",local_dir="/mockvox/pretrained/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch")
# model_dir = snapshot_download("iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",local_dir="/mockvox/pretrained/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch")
model_dir = snapshot_download("iic/punc_ct-transformer_cn-en-common-vocab471067-large",local_dir="/mockvox/pretrained/iic/punc_ct-transformer_cn-en-common-vocab471067-large")