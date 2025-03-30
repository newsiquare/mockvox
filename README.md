# bot

陪护机器人

## 安装
使用SSH连接访问公司[Gitlab仓库](https://gitlab.datainside.com.cn:20443)，需要在您的本地生成 ed25519 密钥：
```bash
ssh-keygen -t ed25519 -C "xxx@xx.com"
```
然后将 id_ed25519.pub 加入到 gitlab 工作台的SSH Keys中：点击头像 -> Edit profile -> SSH Keys -> Add new key

公司Gitlab仓库使用的不是标准SSH端口22, 而是端口20022, 所以您还需要在 {~}\.ssh\config 文件中增加：

```bash
# Windows 的~宿主目录一般位于 C:\Users\your_username目录，~\.ssh\config文件中增加：
Host gitlab.datainside.com.cn
	Port 20022
	User {your login name}
	IdentityFile {~}\.ssh\id_ed25519
```

#### 克隆本项目
```bash
git clone git@gitlab.datainside.com.cn:drz/bot.git
cd bot
```
#### 虚拟环境
```bash
# 创建虚拟环境
conda create -n bot python=3.11 -y
# 激活虚拟环境
conda activate bot
# 安装依赖项(开发环境)
pip install -e .[dev]
```
#### 运行本项目
请确保您的运行环境中已经安装了docker。
```bash
# 运行 docker+redis
./startup_redis.sh
# 检查 redis 运行状态
./check_redis.sh
# 运行 celcery worker
celery -A bot.worker.tasks worker --loglevel=info --concurrency=4
# 打开另一个终端，运行 web server
cd bot
conda activate bot
python bot/main.py
```
