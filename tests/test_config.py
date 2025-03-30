from bot.config import get_config, celery_config

def test_global_config():
    cfg = get_config()
    assert cfg.NAME == 'bot'

def test_celery_config():
    assert hasattr(celery_config, 'broker_url')
    assert isinstance(celery_config.task_publish_retry_policy, dict)