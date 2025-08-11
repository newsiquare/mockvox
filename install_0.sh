#!/bin/bash
set -e  # 發生錯誤就停止

# 1. 建立 Miniconda 目錄
mkdir -p ~/home/miniconda3

# 2. 下載 Miniconda 安裝檔
echo "=== 下載 Miniconda ==="
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    -O ~/home/miniconda3/miniconda.sh

# 3. 安裝 Miniconda
echo "=== 安裝 Miniconda ==="
bash ~/home/miniconda3/miniconda.sh -b -u -p ~/miniconda3

# 4. 移除安裝檔
rm ~/home/miniconda3/miniconda.sh

# 5. 初始化 conda
echo "=== 初始化 conda ==="
source ~/miniconda3/bin/activate
conda init --all

# 6. 建立並啟用 Python 環境
echo "=== 建立 Conda python3.11 環境 > tts ==="
conda create -n tts python=3.11 -y
conda activate tts

# 7. 下載 mockvox 專案
echo "=== 下載 mockvox ==="
git clone https://github.com/mockvox/mockvox.git
cd mockvox

# 8. 安裝套件
echo "=== 安裝套件 ==="
pip install -e .
echo "=== 完成套件安裝 ==="

# 9. 複製環境設定檔
cp .env.sample .env

# 10. 安裝 ffmpeg
echo "=== 安裝 ffmpeg（避免 ffprobe 問題） ==="
sudo apt update && sudo apt install ffmpeg -y
ffmpeg -version
echo "=== 安裝 ox 及開發套件（避免相容性問題） ==="
sudo apt install -y sox libsox-dev
echo "=== 安裝 psmisc（釋放 GPU 記憶體會用到） ==="
sudo apt install -y psmisc

echo "=== 安裝完成 ==="
echo "=== 請重新打開 Terminal 或執行 'source ~/.bashrc' ==="

