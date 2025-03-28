import os
import pytest
from fastapi.testclient import TestClient
from celery.result import AsyncResult
import tempfile
import numpy as np

# 覆盖 Celery 配置为同步模式
os.environ["CELERY_TASK_ALWAYS_EAGER"] = "True"  
os.environ["CELERY_TASK_EAGER_PROPAGATES"] = "True"

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.main import app

@pytest.fixture(scope="session")
def celery_worker():
    """启动同步Worker"""
    with start_worker(
        celery_app,
        pool="solo",
        loglevel="INFO",
        perform_ping_check=False
    ):
        yield

@pytest.fixture
def test_client(celery_worker):  # 依赖celery_worker
    with tempfile.TemporaryDirectory() as tmpdir:
        os.environ["UPLOAD_PATH"] = tmpdir
        os.environ["SLICED_ROOT_PATH"] = os.path.join(tmpdir, "processed")
        with TestClient(app) as client:
            yield client

@pytest.fixture
def valid_wav_file():
    """生成1MB的测试用WAV文件"""
    file_size = 1024 * 1024  # 1MB
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        # 生成随机数据模拟音频文件
        data = np.random.bytes(file_size)
        f.write(data)
        f.seek(0)
        yield f.name
    os.unlink(f.name)

@pytest.fixture
def large_wav_file():
    """生成11MB的测试用WAV文件"""
    file_size = 11 * 1024 * 1024  # 11MB
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(b"\0" * file_size)  # 快速生成大文件
        f.seek(0)
        yield f.name
    os.unlink(f.name)

def test_valid_upload(test_client, valid_wav_file):
    """测试同步任务执行"""
    original_name = os.path.basename(valid_wav_file)
    
    with open(valid_wav_file, "rb") as f:
        response = test_client.post(
            "/upload",
            files={"file": (original_name, f, "audio/wav")}
        )
    
    assert response.status_code == 200
    data = response.json()
    
    # 直接验证任务结果
    assert data["task_id"] is not None
    task = celery_app.AsyncResult(data["task_id"])
    assert task.status == "SUCCESS"
    assert "sliced_file" in task.result["path"]

def test_large_file_upload(test_client, large_wav_file):
    """测试超过大小限制的文件上传"""
    # 获取原始文件名
    original_name = os.path.basename(large_wav_file)
    
    with open(large_wav_file, "rb") as f:
        response = test_client.post(
            "/upload",
            files={"file": (original_name, f, "audio/wav")}
        )
    
    # 验证响应状态
    assert response.status_code == 413
    assert response.json()["detail"] == "文件大小超过限制"
    
    # 验证文件未被保存
    expected_path = os.path.join(os.environ["UPLOAD_PATH"], original_name)
    assert not os.path.exists(expected_path)

def test_invalid_file_type(test_client):
    """测试非WAV文件上传"""
    with tempfile.NamedTemporaryFile(suffix=".mp3") as f:
        f.write(b"fake audio data")
        f.seek(0)
        response = test_client.post(
            "/upload",
            files={"file": ("test.mp3", f, "audio/mpeg")}
        )
    
    assert response.status_code == 400
    assert "不支持的文件格式" in response.json()["detail"]