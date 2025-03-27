# bot

陪护机器人

## 安装

本仓库位于公司[Gitlab仓库](https://gitlab.datainside.com.cn:20443/drz/bot)。如果你没有登录权限，请联系管理员。

要使用SSH连接访问，请先生成 ed25519 密钥：
```bash
ssh-keygen -t ed25519 -C "xxx@xx.com"
```
然后将 id_ed25519.pub 加入到 gitlab 工作台的SSH Keys中：点击头像 -> Edit profile -> SSH Keys -> Add new key

公司Gitlab仓库使用的不是标准SSH端口22, 所以你还需要在 {~}\.ssh\config 文件中增加：

```bash
# Windows 的~宿主目录一般位于 C:\Users\your_username 目录下
Host gitlab.datainside.com.cn
	Port 20022
	User {your login name}
	IdentityFile {~}\.ssh\id_ed25519
```

克隆本项目:
```bash
git clone git@gitlab.datainside.com.cn:drz/bot.git
cd bot
```
虚拟环境
```bash
# 创建虚拟环境
conda create -n bot python=3.11 -y
# 激活虚拟环境
conda activate bot
# 安装依赖项
pip install .
```
