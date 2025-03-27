from config import get_config

cfg = get_config()
redis_url = f"redis://{cfg.REDIS_HOST}:{cfg.REDIS_PORT}/{cfg.REDIS_DB}"

broker_url = redis_url
result_backend = redis_url
task_serializer = 'json'
result_serializer = 'json'
accept_content = ['json']
timezone = "Asia/Shanghai"
enable_utc = True