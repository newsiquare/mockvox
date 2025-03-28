import pytest
import redis
from celery import Celery, states
from celery.contrib.testing.worker import start_worker
from config import get_config, CeleryConfig

cfg = get_config()

class TestTasks:
    """独立的任务容器类，避免循环引用"""
    @staticmethod
    def ping():
        return "pong"
    
    @staticmethod
    def add(x, y):
        return x + y

@pytest.fixture(scope="module")
def celery_app():
    """创建完全隔离的 Celery 测试环境"""
    # 创建独立应用实例
    app = Celery("test_worker")
    app.config_from_object(CeleryConfig)
    
    # 手动注册任务（绕过装饰器系统）
    app.task(name="celery.ping", staticmethod=True)(TestTasks.ping)
    app.task(name="test_add", staticmethod=True)(TestTasks.add)
    
    # 启动Worker（禁用所有自动检查）
    with start_worker(
        app,
        pool="solo",
        loglevel="INFO",
        perform_ping_check=False
    ):
        # 手动验证任务注册
        assert "celery.ping" in app.tasks
        assert app.conf.task_serializer == "json"
        yield app

def test_task_execution(celery_app):
    """验证任务执行"""
    # 直接通过应用实例提交任务
    result = celery_app.send_task("test_add", args=(3,4))
    assert result.get(timeout=5) == 7

def test_redis_connection():
    """验证Redis连接"""
    r = redis.Redis(
        host=cfg.REDIS_HOST,
        port=cfg.REDIS_PORT,
        db=cfg.REDIS_DB,
        password=cfg.REDIS_PASSWORD
    )
    assert r.ping(), "Redis连接失败"

