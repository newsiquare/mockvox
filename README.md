# bot

陪护机器人

## 安装
使用SSH连接访问公司[Gitlab仓库](https://gitlab.datainside.com.cn:20443)，需要在您的本地生成 ed25519 密钥：
```bash
ssh-keygen -t ed25519 -C "xxx@xx.com"
```
然后将 id_ed25519.pub 加入到 gitlab 工作台的SSH Keys中：点击头像 -> Edit profile -> SSH Keys -> Add new key

公司Gitlab仓库使用的不是标准SSH端口22, 而是端口20022, 所以您还需要在 ~\\.ssh\config 文件中增加：

```bash
# Windows 的~宿主目录一般位于 C:\Users\your_username目录，~\.ssh\config文件中增加：
Host gitlab.datainside.com.cn
	Port 20022
	User {your login name}
	IdentityFile {~}\.ssh\id_ed25519
```

#### 克隆本项目
```bash
git clone git@gitlab.datainside.com.cn:fakevoi/bot.git
cd bot
```
#### 虚拟环境
```bash
# 创建虚拟环境
conda create -n bot python=3.11 -y
# 激活虚拟环境
conda activate bot
# 安装依赖项(开发环境)
pip install -e .[dev,audio]
# 安装依赖项(生产环境)
pip install .[audio]

```
## 运行本项目
#### 安装ffmpeg
安装ffmpeg(这里仅提供了ubuntu安装脚本)。
```bash
# 安装ffmpeg
sudo apt update
sudo apt install ffmpeg
## 检查安装
ffmpeg -version
```
#### 安装预训练模型
```bash
# 语音降噪模型
modelscope download damo/speech_frcrn_ans_cirm_16k
# 语音识别模型
modelscope download iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch
# 标点恢复模型
modelscope download iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch
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
celery -A bot.worker.worker worker --loglevel=info --concurrency=4
# 打开另一个终端，运行 web server
cd bot
conda activate bot
python bot/main.py
```
