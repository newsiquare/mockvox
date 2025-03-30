from celery import Celery
from bot.config import get_config, celery_config
import os

cfg = get_config()
UPLOAD_PATH = cfg.UPLOAD_PATH
os.makedirs(cfg.SLICED_ROOT_PATH, exist_ok=True)

app = Celery("worker")
app.config_from_object(celery_config)

@app.task(name="process_file", bind=True)
def process_file_task(self, file_name: str):
    try:
        stem, _ = os.path.splitext(file_name)
        file_path = os.path.join(UPLOAD_PATH, file_name)
        
        # 处理逻辑示例
        sliced_path = os.path.join(
            cfg.SLICED_ROOT_PATH, 
            stem
        )
        os.makedirs(sliced_path, exist_ok=True)
        
        # TODO: 添加实际处理逻辑
        sliced_file = os.path.join(sliced_path, file_name)
        with open(file_path, "rb") as src, open(sliced_file, "wb") as dst:
            while chunk := src.read(1024 * 1024):    # 1Mb chunks
                dst.write(chunk)
        
        return {"status": "success", "path": sliced_file}
    
    except Exception as e:
        # 错误重试逻辑
        raise self.retry(exc=e, countdown=60, max_retries=3)
    
# 在 tasks.py 底部添加
if __name__ == "__main__":
    print("当前生效配置:")
    print("Broker URL:", app.conf.broker_url)
    print("Result Backend:", app.conf.result_backend)