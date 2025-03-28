import pytest
import redis
from config import get_config

cfg = get_config()

def test_redis_connection():
    pool = redis.ConnectionPool(
        host=cfg.REDIS_HOST,
        port=cfg.REDIS_PORT,
        db=cfg.REDIS_DB,
        password=cfg.REDIS_PASSWORD,
        decode_responses = True
    )
    r = redis.Redis(connection_pool=pool)
    assert r.ping()