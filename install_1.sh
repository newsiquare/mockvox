#!/bin/bash
set -e  # 發生錯誤就停止

# 建立 pretrained 目錄
mkdir -p pretrained

echo "🔧 下載 GPT-SoVITS Core Components..."
git clone https://huggingface.co/lj1995/GPT-SoVITS.git ./pretrained/GPT-SoVITS

echo "🗣️ 下載 Voice Processing Suite 模型..."
modelscope download --model 'damo/speech_frcrn_ans_cirm_16k' \
    --local_dir './pretrained/damo/speech_frcrn_ans_cirm_16k'  # Denoise

modelscope download --model 'iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch' \
    --local_dir './pretrained/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch'  # Mandarin ASR

modelscope download --model 'iic/speech_fsmn_vad_zh-cn-16k-common-pytorch' \
    --local_dir './pretrained/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch'  # VAD

modelscope download --model 'iic/punc_ct-transformer_cn-en-common-vocab471067-large' \
    --local_dir './pretrained/iic/punc_ct-transformer_cn-en-common-vocab471067-large'  # Punctuation restoration

git clone https://huggingface.co/alextomcat/G2PWModel.git ./pretrained/G2PWModel  # Grapheme-to-phoneme

echo "🌐 下載 Multilingual Extensions (可選)..."
modelscope download --model 'iic/speech_UniASR_asr_2pass-cantonese-CHS-16k-common-vocab1468-tensorflow1-online' \
    --local_dir './pretrained/iic/speech_UniASR_asr_2pass-cantonese-CHS-16k-common-vocab1468-tensorflow1-online'  # Cantonese ASR

git clone https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2.git ./pretrained/nvidia/parakeet-tdt-0.6b-v2  # English ASR

git clone https://huggingface.co/FacebookAI/roberta-large.git ./pretrained/FacebookAI/roberta-large  # English BERT

modelscope download --model 'iic/speech_UniASR_asr_2pass-ja-16k-common-vocab93-tensorflow1-offline' \
    --local_dir './pretrained/iic/speech_UniASR_asr_2pass-ja-16k-common-vocab93-tensorflow1-offline'  # Japanese ASR

git clone https://huggingface.co/tohoku-nlp/bert-large-japanese-v2.git ./pretrained/tohoku-nlp/bert-large-japanese-v2  # Japanese BERT

modelscope download --model 'iic/speech_UniASR_asr_2pass-ko-16k-common-vocab6400-tensorflow1-online' \
    --local_dir './pretrained/iic/speech_UniASR_asr_2pass-ko-16k-common-vocab6400-tensorflow1-online'  # Korean ASR

git clone https://huggingface.co/klue/roberta-large.git ./pretrained/klue/roberta-large  # Korean BERT

echo "✅ 所有模型下載完成！"
