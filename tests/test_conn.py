import pytest
import redis
from celery import Celery
import time
from config import get_config, CeleryConfig

cfg = get_config()

@pytest.fixture(scope="module")
def celery_app():
    app = Celery("test_worker")
    app.config_from_object(CeleryConfig)
    
    # 添加测试任务
    @app.task(name="test_add")
    def add(x, y):
        return x + y
    
    # 使用独立进程启动Worker
    worker = app.Worker(
        hostname="tester@%%h",
        pool="solo",
        loglevel="DEBUG",  # 调试关键
        without_heartbeat=True,
        without_mingle=True,
        without_gossip=True,
        background=True 
    )
    worker.start()  
    
    # 等待Worker就绪
    start = time.time()
    while not worker.consumer.ready() and time.time() - start < 10:
        time.sleep(0.1)
    
    yield app
    
    # 强制终止Worker
    worker.stop(in_sighandler=True)
    worker.join() 

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
    
    # 添加结果检查重试机制
    max_wait = 5  
    start = time.time()
    
    while not result.ready():
        if time.time() - start > max_wait:
            assert False, f"任务超时未完成，最后状态: {result.state}"
        time.sleep(0.5)
    
    assert result.get() == 7
    
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
    
    # 动态跟踪状态
    for _ in range(10):  # 最多检查10次
        state = result.state
        if state != "PENDING":
            break
        time.sleep(0.5)
    else:
        assert False, "任务状态未更新"
    
    assert state == "RETRY", "未触发重试机制"
    assert result.traceback, "缺少错误堆栈"