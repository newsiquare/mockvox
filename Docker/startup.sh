#!/bin/bash

set -e  # 出现错误立即退出
set -o pipefail  # 管道命令错误退出

chmod +x /mockvox/Docker/generalDownload.sh
/mockvox/Docker/generalDownload.sh
python /mockvox/Docker/generalDownload.py
# 安装中文模型文件
if [ "$MODEL_TYPE" == "full" ]; then
	chmod +x /mockvox/Docker/englishDownload.sh
	/mockvox/Docker/englishDownload.sh
	python /mockvox/Docker/cantoneseDownload.py
	chmod +x /mockvox/Docker/japaneseDownload.sh
	/mockvox/Docker/japaneseDownload.sh
	python /mockvox/Docker/japaneseDownload.py
	chmod +x /mockvox/Docker/koreanDownload.sh
	/mockvox/Docker/koreanDownload.sh
	python /mockvox/Docker/koreanDownload.py
elif [ "$MODEL_TYPE" == "en" ]; then
	chmod +x /mockvox/Docker/englishDownload.sh
	/mockvox/Docker/englishDownload.sh
elif [ "$MODEL_TYPE" == "can" ]; then
	python /mockvox/Docker/cantoneseDownload.py
elif [ "$MODEL_TYPE" == "ja" ]; then
	chmod +x /mockvox/Docker/japaneseDownload.sh
	/mockvox/Docker/japaneseDownload.sh
	python /mockvox/Docker/japaneseDownload.py
elif [ "$MODEL_TYPE" == "ko" ]; then
	chmod +x /mockvox/Docker/koreanDownload.sh
	/mockvox/Docker/koreanDownload.sh
	python /mockvox/Docker/koreanDownload.py
fi


cd /mockvox
# 确保文件存在
touch .env.sample
cp .env.sample .env
# 删除文件中的redis密码配置
sed -i '/^REDIS_PASSWORD=/d' .env 2>/dev/null
echo "REDIS_PASSWORD=$REDIS_PASSWORD" >> .env
if [ "$REDIS_PORT" != "" ]; then
	sed -i '/^REDIS_PORT=/d' .env 2>/dev/null
	echo "REDIS_PORT=$REDIS_PORT" >> .env
fi

mkdir -p /mockvox/log
nohup celery -A src.mockvox.worker.worker worker --loglevel=info --pool=prefork --concurrency=1 > log/celery.log 2>&1 &
python src/mockvox/main.py