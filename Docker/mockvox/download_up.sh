#!/bin/bash


chmod +x /mockvox/Docker/nomalDownload.sh
/mockvox/Docker/nomalDownload.sh
python /mockvox/Docker/nomalDownload.py
# 安装中文模型文件
if [ "$MODEL_TYPE" == "full" ]; then
	chmod +x /mockvox/Docker/englishDownload.sh
	/mockvox/Docker/engliseDownload.sh
	python /mockvox/Docker/cantoneseDownload.py
	chmod +x /mockvox/Docker/japaneseDownload.sh
	/mockvox/Docker/japaneseDownload.sh
	python /mockvox/Docker/japaneseDownload.py
	chmod +x /mockvox/Docker/koreanDownload.sh
	/mockvox/Docker/koreanDownload.sh
	python /mockvox/Docker/koreanDownload.py
elif [ "$MODEL_TYPE" == "english" ]; then
	chmod +x /mockvox/Docker/englishDownload.sh
	/mockvox/Docker/engliseDownload.sh
elif [ "$MODEL_TYPE" == "cantonese" ]; then
	python /mockvox/Docker/cantoneseDownload.py
elif [ "$MODEL_TYPE" == "japanese" ]; then
	chmod +x /mockvox/Docker/japaneseDownload.sh
	/mockvox/Docker/japaneseDownload.sh
	python /mockvox/Docker/japaneseDownload.py
elif [ "$MODEL_TYPE" == "korean" ]; then
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
if [ "$REDIS_PORT" != ""]; then
	sed -i '/^REDIS_PORT=/d' .env 2>/dev/null
	echo "REDIS_PORT=$REDIS_PORT" >> .env

nohup celery -A src.bot.worker.worker worker --loglevel=info --pool=prefork --concurrency=1 > log/celery.log 2>&1 &
nohup python src/bot/main.py > log/main.log 2>&1 & 
