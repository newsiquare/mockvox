import pytest
from config import get_config, CeleryConfig

def test_global_config():
    cfg = get_config()
    assert cfg.NAME == 'bot'

def test_celery_config():
    assert hasattr(CeleryConfig, 'broker_url')
    assert isinstance(CeleryConfig.task_publish_retry_policy, dict)