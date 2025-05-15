# bot

本项目旨在打造一个可以社区化运作的语言合成&语音克隆平台。  
本项目改造自 [GPT_SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)，提供和GPT_SoVITS相同流程的语音合成&语音克隆功能。

主要的改造点有:

1. 去掉Web端，改用命令行方式;
2. 改为用 celery 管理后台异步任务。为此，去掉所有的 torch 分布式训练逻辑，改为由 celery 调度后台的训练、推理进程；
3. 英语ASR模型不用Faster-Whisper(存在cuda+nvidia driver版本兼容问题),改用 [Nvidia Parakeet TDT 0.6B V2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2)；
4. 日、韩ASR模型改用 [iic/speech_UniASR_asr_2pass-ja-16k-common-vocab93-tensorflow1-offline](https://www.modelscope.cn/models/iic/speech_UniASR_asr_2pass-ja-16k-common-vocab93-tensorflow1-offline), [iic/speech_UniASR_asr_2pass-ko-16k-common-vocab6400-tensorflow1-offline](https://www.modelscope.cn/models/iic/speech_UniASR_asr_2pass-ko-16k-common-vocab6400-tensorflow1-offline)
5. 代码优化；

## 克隆本项目

```bash
git clone git@gitlab.datainside.com.cn:fakevoi/bot.git
cd bot
```

## 创建虚拟环境

```bash
# 创建虚拟环境
conda create -n bot python=3.11 -y
# 激活虚拟环境
conda activate bot
# 安装依赖项(生产环境)
pip install .
# 安装依赖项(开发环境)
pip install -e .[dev]  
```

## 运行本项目

### 安装ffmpeg

安装ffmpeg(这里仅提供了ubuntu安装脚本)。

```bash
# 安装ffmpeg
sudo apt update
sudo apt install ffmpeg
## 检查安装
ffmpeg -version
```

#### 下载预训练模型

**注意** 保持当前目录为项目根目录。

```bash
# ------------------------------------------------ 数据预处理模型 ----------------------------------------------------------
# 语音降噪模型
modelscope download --model 'damo/speech_frcrn_ans_cirm_16k' --local_dir './pretrained/damo/speech_frcrn_ans_cirm_16k'
# 普通话ASR模型
modelscope download --model 'iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch' --local_dir './pretrained/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch'
# 语音端点检测
modelscope download --model 'iic/speech_fsmn_vad_zh-cn-16k-common-pytorch' --local_dir './pretrained/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch'
# 标点恢复模型
modelscope download --model 'iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch' --local_dir './pretrained/iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch'
# 词转音素
git clone https://huggingface.co/alextomcat/G2PWModel.git ./pretrained/G2PWModel

# [可选] 粤语ASR模型
modelscope download --model 'iic/speech_UniASR_asr_2pass-cantonese-CHS-16k-common-vocab1468-tensorflow1-online' --local_dir './pretrained/iic/speech_UniASR_asr_2pass-cantonese-CHS-16k-common-vocab1468-tensorflow1-online'
# [可选] 英语ASR模型
git clone https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2.git ./pretrained/nvidia/parakeet-tdt-0.6b-v2
# [可选] 日语ASR模型
modelscope download --model 'iic/speech_UniASR_asr_2pass-ja-16k-common-vocab93-tensorflow1-offline'  --local_dir './pretrained/iic/speech_UniASR_asr_2pass-ja-16k-common-vocab93-tensorflow1-offline'
# [可选] 韩语ASR模型
modelscope download --model 'iic/speech_UniASR_asr_2pass-ko-16k-common-vocab6400-tensorflow1-offline' --local_dir './pretrained/iic/speech_UniASR_asr_2pass-ko-16k-common-vocab6400-tensorflow1-offline'

# ------------------------------------------------ GPT-SoVITS 模型 --------------------------------------------------------
# GPT-SoVITS:  
git clone https://huggingface.co/lj1995/GPT-SoVITS.git ./pretrained/GPT-SoVITS
```

本项目需要在docker环境中运行redis, 请确保您的运行环境中已经安装了docker。

```bash
# 复制环境变量文件
cp .env.sample .env
# 运行 docker+redis (如果是第一次运行，需要从 docker镜像库拉取redis镜像，请确保您的网络能够正常拉取docker镜像。)
chmod +x startup_redis.sh
./startup_redis.sh
# 检查 redis 运行状态
chmod +x check_redis.sh
./check_redis.sh
# 运行 celcery worker
celery -A src.bot.worker.worker worker --loglevel=info --pool=prefork --concurrency=1
# 打开另一个终端，运行 web server
cd bot
conda activate bot
python src/bot/main.py
```
