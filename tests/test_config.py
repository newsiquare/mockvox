import pytest
from config import get_config
def test_global_config():
    cfg = get_config()
    assert cfg.NAME == 'bot'
