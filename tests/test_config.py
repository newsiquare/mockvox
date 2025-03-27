import pytest
from bot import get_settings
def test_global_config():
    settings = get_settings()
    assert settings.NAME == 'bot'
