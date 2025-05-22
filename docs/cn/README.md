<div align="center">

<h1>🎤 MockVox</h1>

✨ 强大的少样本语音合成与语音克隆后台 ✨<br><br>

[**🇬🇧 English**](../../README.md) | **🇨🇳 中文简体**

</div>

---

## 🚀 介绍

本项目旨在打造一个可以社区化运作的语音合成&语音克隆平台。  
本项目改造自 [GPT_SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)，提供和GPT_SoVITS相同流程的语音合成&语音克隆功能。  

🌟 **核心功能**：

1. **🎧 零样本文本到语音 (TTS)**: 输入 5 秒的声音样本，即刻体验文本到语音转换
2. **🧠 少样本 TTS**: 仅需 1 分钟的训练数据即可微调模型，提升声音相似度和真实感
3. **🌍 跨语言支持**: 支持英语、日语、韩语、粤语和中文的多语言推理

🔧 **主要改造点**：

1. **🖥️ 命令行交互**：去掉Web端，改用更灵活的命令行方式 [《命令行用户指南》](./cli.md)
2. **🏭 分布式架构**：基于 Celery 实现多进程异步任务调度，支持高并发训练/推理
3. **⚡ 训练优化**：弃用 Pytorch Lightning，采用原生 Torch 训练方式
4. **🔊 ASR 升级**：英语模型改用 NVIDIA Parakeet，日韩模型采用 ModelScope 最新方案
5. **🌐 多语言适配**：为不同语言配置专用 BERT 特征提取模型
6. **📦 工程优化**：代码重构与性能提升

---

## 📥 安装

### 克隆本项目

```bash
git clone git@gitlab.datainside.com.cn:fakevoi/bot.git
cd bot
```

---

## 🚀 运行

### 🖥️ 本地运行

#### 1. 创建虚拟环境

```bash
🐍 创建 Python 虚拟环境
conda create -n bot python=3.10 -y
conda activate bot

📦 安装依赖
pip install .          # 生产环境
pip install -e .[dev]  # 开发环境
```

#### 2. 安装 FFmpeg

```bash
🎬 Ubuntu 安装脚本
sudo apt update && sudo apt install ffmpeg
ffmpeg -version  # 验证安装
```

#### 3. 下载预训练模型

```bash
🔧 GPT-SoVITS核心组件
git clone https://huggingface.co/lj1995/GPT-SoVITS.git ./pretrained/GPT-SoVITS

🗣️ 语音处理全家桶
modelscope download --model 'damo/speech_frcrn_ans_cirm_16k' --local_dir './pretrained/damo/speech_frcrn_ans_cirm_16k' #降噪
modelscope download --model 'iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch' --local_dir './pretrained/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch' #普通话ASR
modelscope download --model 'iic/speech_fsmn_vad_zh-cn-16k-common-pytorch' --local_dir './pretrained/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch' #端点检测
modelscope download --model 'iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch' --local_dir './pretrained/iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch' #标点恢复
git clone https://huggingface.co/alextomcat/G2PWModel.git ./pretrained/G2PWModel #词转音素


🌐 多语言扩展包（可选）
modelscope download --model 'iic/speech_UniASR_asr_2pass-cantonese-CHS-16k-common-vocab1468-tensorflow1-online' --local_dir './pretrained/iic/speech_UniASR_asr_2pass-cantonese-CHS-16k-common-vocab1468-tensorflow1-online' #粤语ASR
git clone https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2.git ./pretrained/nvidia/parakeet-tdt-0.6b-v2 #英语ASR
modelscope download --model 'iic/speech_UniASR_asr_2pass-ja-16k-common-vocab93-tensorflow1-offline'  --local_dir './pretrained/iic/speech_UniASR_asr_2pass-ja-16k-common-vocab93-tensorflow1-offline' #日语ASR
modelscope download --model 'iic/speech_UniASR_asr_2pass-ko-16k-common-vocab6400-tensorflow1-offline' --local_dir './pretrained/iic/speech_UniASR_asr_2pass-ko-16k-common-vocab6400-tensorflow1-offline' #韩语ASR
git clone https://huggingface.co/FacebookAI/roberta-large.git ./pretrained/FacebookAI/roberta-large #英语BERT
git clone https://huggingface.co/tohoku-nlp/bert-base-japanese-v3.git ./pretrained/tohoku-nlp/bert-base-japanese-v3 #日语BERT
git clone https://huggingface.co/klue/bert-base.git ./pretrained/klue/bert-base #韩语BERT
```

#### 4. 启动服务

```bash
🐳 Redis 容器
chmod +x startup_redis.sh && ./startup_redis.sh
chmod +x check_redis.sh && ./check_redis.sh  # ✅ 状态检查

⚙️ Celery 工作节点
nohup celery -A src.bot.worker.worker worker --loglevel=info --pool=prefork --concurrency=1 &

🌐 Web 服务
nohup python src/bot/main.py &
```

### 🐳 Docker 容器运行

```bash
cd Docker
docker-compose up # 🚢 一键启动全栈服务
```

---

## 🔍 扩展说明

- 📁 所有模型文件默认存储在 `./pretrained` 目录
- ⚠️ 首次运行需下载约 15GB 的模型文件
- 🔄 可通过修改 `.env` 文件调整服务配置
- 📚 完整命令行指南请参阅 [CLI 文档](./cli.md)

---

<div align="center">
  <sub>Built with ❤️ by MockVox Team | 📧 Contact: dev@mockvox.cn</sub>
</div>