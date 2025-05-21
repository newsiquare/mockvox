#!/bin/bash

# 确保 .env 文件存在
touch .env

# 清理旧的环境变量定义（避免重复）
sed -i '/^REDIS_PASSWORD=/d' .env 2>/dev/null
sed -i '/^HOST_PORT=/d' .env 2>/dev/null

# 生成随机密码（16位字母数字组合）
random_password=$(openssl rand -base64 32 | tr -dc 'a-zA-Z0-9' | head -c 16)
echo "REDIS_PASSWORD=${random_password}" >> .env

# 设置默认端口（如果未在 .env 中定义）
if ! grep -q 'REDIS_PORT=' .env; then
  echo "REDIS_PORT=6379" >> .env
fi
# 启动容器
docker-compose --env-file .env up -d

# 显示连接信息
echo "Redis 已启动："
echo "- 端口: $(grep 'REDIS_PORT=' .env | cut -d '=' -f2)"
echo "- 密码: $(grep 'REDIS_PASSWORD=' .env | cut -d '=' -f2)"