import pytest
import redis
from celery import Celery
import time
from config import get_config, CeleryConfig

cfg = get_config()

# 创建测试用 Celery 应用
@pytest.fixture(scope="module")
def celery_app():
    app = Celery("test_worker")
    app.config_from_object(CeleryConfig)
    
    # 添加测试任务
    @app.task(name="test_add")
    def add(x, y):
        return x + y
    
    # 启动测试worker（单进程模式）
    worker = app.Worker(
        hostname="tester@%%h",
        pool="solo",
        loglevel="WARNING",
        without_heartbeat=True,
        without_mingle=True,
        without_gossip=True
    )
    worker.start()
    yield app
    worker.stop()

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
        password=cfg.REDIS_PASSWORD,  
        decode_responses=True
    )
    assert r.ping(), "Redis连接失败"

def test_task_execution(celery_app):
    """验证任务提交与结果存储"""
    # 提交任务
    result = celery_app.tasks["test_add"].delay(3, 4)
    
    # 等待任务完成（最大等待5秒）
    start = time.time()
    while not result.ready() and time.time() - start < 5:
        time.sleep(0.1)
    
    assert result.ready(), "任务超时未完成"
    assert result.get() == 7, "任务结果错误"
    
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
    # 定义会失败的任务
    @celery_app.task(bind=True, name="test_retry", max_retries=3)
    def fail_task(self):
        try:
            raise ValueError("模拟错误")
        except ValueError as exc:
            raise self.retry(exc=exc, countdown=0.1)
    
    # 提交任务
    result = fail_task.delay()
    
    # 等待重试完成
    start = time.time()
    while result.state == "PENDING" and time.time() - start < 5:
        time.sleep(0.1)

    assert result.state == "RETRY", "未触发重试机制"
    assert result.traceback, "缺少错误堆栈"