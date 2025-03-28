#!/bin/sh
set -e

# 加载项目环境变量（挂载到/app/.env）
if [ -f /app/.env ]; then
    export $(grep -v '^#' /app/.env | xargs)
fi

# 生成最终配置文件
envsubst < /app/redis/redis.conf.template > /app/redis/redis.conf

# 启动Redis
exec redis-server /app/redis.conf \
  --requirepass "$REDIS_PASSWORD" \ 
  --maxmemory "$REDIS_MEMORY_LIMIT"
