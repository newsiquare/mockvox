import os
import pytest
from fastapi.testclient import TestClient
from main import app
from celery.result import AsyncResult
import tempfile
import numpy as np

@pytest.fixture(scope="module")
def test_client():
    # 使用临时目录存储测试文件
    with tempfile.TemporaryDirectory() as tmpdir:
        # 覆盖配置中的上传路径
        os.environ["UPLOAD_PATH"] = tmpdir
        os.environ["SLICED_ROOT_PATH"] = os.path.join(tmpdir, "processed")
        yield TestClient(app)

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
    """测试正常文件上传流程"""
    # 获取原始文件名
    original_name = os.path.basename(valid_wav_file)
    
    with open(valid_wav_file, "rb") as f:
        response = test_client.post(
            "/upload",
            files={"file": (original_name, f, "audio/wav")}
        )
    
    # 验证响应状态
    assert response.status_code == 200
    
    # 验证响应数据格式
    response_data = response.json()
    assert "filename" in response_data
    assert "task_id" in response_data
    assert response_data["file_size"] == 1024 * 1024
    
    # 验证文件保存路径
    saved_path = os.path.join(os.environ["UPLOAD_PATH"], response_data["filename"])
    assert os.path.exists(saved_path)
    assert os.path.getsize(saved_path) == 1024 * 1024
    
    # 验证Celery任务状态
    task = AsyncResult(response_data["task_id"])
    assert task.status in ("PENDING", "RECEIVED", "STARTED")

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