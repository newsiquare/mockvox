import pytest
import redis
from celery import Celery, states
from celery.contrib.testing.worker import start_worker
import time
from config import get_config, CeleryConfig

cfg = get_config()

# 自定义测试用ping任务（避免使用celery.contrib.testing的冲突实现）
@shared_task(name="celery.ping")  # 必须使用此名称
def ping():
    return "pong"

@pytest.fixture(scope="module")
def celery_app():
    """创建隔离的测试Celery应用"""
    app = Celery("test_worker")
    app.config_from_object(CeleryConfig)
    
    # 手动添加测试任务
    app.register_task(ping)  # 注册自定义ping任务
    
    # 添加用户测试任务
    @app.task(name="test_add")
    def add(x, y):
        return x + y
    
    # 启动Worker
    with start_worker(
        app,
        pool="solo",
        loglevel="INFO",
        perform_ping_check=False  # 禁用自动ping检查
    ):
        # 手动执行ping验证
        result = ping.delay()
        assert result.get(timeout=5) == "pong"
        yield app

def test_config_loaded(celery_app):
    """验证配置项是否正确加载"""
    assert celery_app.conf.broker_url == f"redis://:{cfg.REDIS_PASSWORD}@{cfg.REDIS_HOST}:{cfg.REDIS_PORT}/{cfg.REDIS_DB}"
    assert celery_app.conf.result_backend == celery_app.conf.broker_url
    assert celery_app.conf.task_serializer == "json"

def test_redis_connection():
    """验证Redis连接"""
    r = redis.Redis(
        host=cfg.REDIS_HOST,
        port=cfg.REDIS_PORT,
        db=cfg.REDIS_DB,
        password=cfg.REDIS_PASSWORD
    )
    assert r.ping(), "Redis连接失败"

def test_task_execution(celery_app):
    """验证任务提交与结果存储"""
    result = celery_app.send_task("test_add", args=(3, 4))
    assert result.get(timeout=10) == 7
    
    # 验证结果持久化
    r = redis.Redis(
        host=cfg.REDIS_HOST,
        port=cfg.REDIS_PORT,
        db=cfg.REDIS_DB,
        password=cfg.REDIS_PASSWORD
    )
    assert r.exists(f"celery-task-meta-{result.id}"), "结果未持久化"

def test_retry_policy(celery_app):
    """验证任务重试策略"""
    @celery_app.task(bind=True, name="test_retry", max_retries=3)
    def fail_task(self):
        raise self.retry(countdown=0.1, max_retries=3)
    
    # 提交并捕获重试
    result = fail_task.delay()
    
    # 等待状态变更
    for _ in range(10):
        if result.state == states.RETRY:
            break
        time.sleep(0.5)
    else:
        assert False, "未触发重试机制"
    
    assert result.traceback, "缺少错误堆栈"