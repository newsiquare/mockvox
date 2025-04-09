import os
import time
import pytest
from fastapi.testclient import TestClient
from celery.result import AsyncResult
import numpy as np
from bot.config import BASE_DIR, get_config, UPLOAD_PATH, ASR_PATH

cfg = get_config()

# 设置测试文件
TEST_FILE = os.path.join(BASE_DIR, 'tests/liuyx.WAV')

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from bot.main import app

@pytest.fixture(scope="module")
def test_client():
    with TestClient(app) as client:
        yield client

def generate_test_file(file_size: int) -> bytes:
    """生成指定大小的随机文件内容"""
    return np.random.bytes(file_size)

def wait_for_task_completion(task_id: str, timeout: int = 30) -> dict:
    """等待任务完成并返回最终状态"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        task = AsyncResult(task_id)
        if task.status == "SUCCESS":
            return {"status": task.status, "result": task.result}
        if task.status == "FAILURE":
            return {"status": task.status, "error": str(task.result)}
        time.sleep(0.5)
    return {"status": "TIMEOUT"}

def test_valid_upload_e2e(test_client):
    """端到端测试：有效文件上传全流程"""
    # 1. 读取测试文件
    file_content = open(TEST_FILE, 'rb').read()
    
    # 2. 发送上传请求
    response = test_client.post(
        "/upload",
        files={"file": ("valid.wav", file_content, "audio/wav")}
    )
    
    # 3. 验证接口响应
    response_data = response.json()
    assert response.status_code == 200
    assert "task_id" in response_data
    assert "file_name" in response_data
    
    # 4. 验证文件存储
    saved_path = os.path.join(UPLOAD_PATH, response_data["file_name"])
    assert os.path.exists(saved_path)
    
    # 5. 等待任务完成
    task_info = wait_for_task_completion(response_data["task_id"])
    assert task_info["status"] == "SUCCESS"
    stem, _ = os.path.splitext(response_data["file_name"])
    assert stem in task_info["result"]["path"]
    
    # 6. 验证结果存储
    processed_path = os.path.join(ASR_PATH, task_info["result"]["path"])
    assert os.path.exists(processed_path)

def test_large_file_rejection_e2e(test_client):
    """端到端测试：大文件拦截验证"""
    # 1. 生成11MB测试文件
    large_file = generate_test_file(110 * 1024 * 1024)
    
    # 2. 发送上传请求
    response = test_client.post(
        "/upload",
        files={"file": ("large.wav", large_file, "audio/wav")}
    )
    
    # 3. 验证接口拦截
    assert response.status_code == 413
    assert "文件大小超过限制" in response.json()["detail"]
