#!/bin/bash

set -eo pipefail  # 遇到错误立即退出

# 定义路径常量
ENV_FILE=".env"
REDIS_CONF_TEMPLATE="redis/redis.conf.template"
REDIS_DATA_PATH="redis/data"

# 生成加密密码（避免特殊字符）
generate_password() {
  # 使用 urandom 避免 base64 中的特殊符号
  export REDIS_PASSWORD=$(tr -dc 'A-Za-z0-9' </dev/urandom | head -c 32)
  if [ -z "$REDIS_PASSWORD" ]; then
    echo "密码生成失败！" >&2
    exit 1
  fi

  # 更新.env文件
  if grep -q "REDIS_PASSWORD=" "$ENV_FILE"; then
    sed -i.bak "/REDIS_PASSWORD=/c\REDIS_PASSWORD=$REDIS_PASSWORD" "$ENV_FILE"
  else
    echo "REDIS_PASSWORD=$REDIS_PASSWORD" >> "$ENV_FILE"
  fi
  rm -f "${ENV_FILE}.bak" 2>/dev/null || true
}

# 验证目录结构
validate_paths() {
  [ -f "$REDIS_CONF_TEMPLATE" ] || { echo "缺少Redis模板文件!"; exit 1; }
  mkdir -p "$REDIS_DATA_PATH"
  chmod 700 "$REDIS_DATA_PATH"
}

# 加载环境变量
load_env() {
  [ -f "$ENV_FILE" ] || { echo "缺少.env文件!"; exit 1; }
  export $(grep -v '^#' "$ENV_FILE" | xargs) >/dev/null 2>&1 || true
}

# 清理旧容器
cleanup() {
  docker rm -f bot-redis >/dev/null 2>&1 || true
}

main() {
  echo "正在初始化Redis服务..."
  
  cleanup
  generate_password
  validate_paths
  load_env

  echo "生成的Redis密码: ${REDIS_PASSWORD:0:4}******${REDIS_PASSWORD: -4}"

  # 运行Docker容器
  docker run -d \
    --name bot-redis \
    -p "${REDIS_PORT:-6380}:${REDIS_PORT:-6380}" \
    -v "$(pwd)/$REDIS_DATA_PATH:/data" \
    -v "$(pwd)/$REDIS_CONF_TEMPLATE:/app/redis.conf.template:ro" \
    -e "REDIS_PASSWORD=$REDIS_PASSWORD" \
    -e "REDIS_PORT=${REDIS_PORT:-6380}" \
    -e "REDIS_MEMORY_LIMIT=${REDIS_MEMORY_LIMIT}" \
    --restart unless-stopped \
    redis:7.0-alpine \
    sh -c "apk add gettext && envsubst < /app/redis.conf.template > /app/redis.conf && redis-server /app/redis.conf"

  echo "Redis服务已启动!"
  echo "连接命令:"
  echo "docker exec -it bot-redis redis-cli -h 127.0.0.1 -p ${REDIS_PORT} -a $REDIS_PASSWORD"
}

main "$@"