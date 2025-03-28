#!/bin/bash

source .env

echo "=== 配置验证 ==="
docker exec bot-redis grep -E "port|maxmemory" /app/redis.conf

echo -e "\n=== 容器内连接测试 ==="
docker exec bot-redis \
  redis-cli -h 127.0.0.1 -p $REDIS_PORT -a "$REDIS_PASSWORD" ping