from .config import get_config
cfg = get_config()

class CeleryConfig:
    # 连接配置
    broker_url = f"redis://:{cfg.REDIS_PASSWORD}@{cfg.REDIS_HOST}:{cfg.REDIS_PORT}/{cfg.REDIS_DB_BROKER}"
    result_backend = f"redis://:{cfg.REDIS_PASSWORD}@{cfg.REDIS_HOST}:{cfg.REDIS_PORT}/{cfg.REDIS_DB_RESULT}"
    
    # 结果保留24小时
    result_expires = 60 * 60 * 24
    # 保持结果持久化直到过期
    result_persistent = True
    
    # 序列化配置
    task_serializer = 'json'
    result_serializer = 'json'
    accept_content = ['json']
    
    # 时区配置
    timezone = "Asia/Shanghai"
    enable_utc = True
    
    # 任务发布重试策略
    task_publish_retry = True
    task_publish_retry_policy = {
        'max_retries': 3,
        'interval_start': 0.2,
        'interval_step': 0.3,
        'interval_max': 1.0,
    }
    
    # 其他高级配置
    worker_prefetch_multiplier = 1  # 控制并发性能
    task_acks_late = True  # 确保任务不丢失

celery_config = CeleryConfig()