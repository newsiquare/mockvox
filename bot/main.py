import os
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.requests import Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from celery.result import AsyncResult

from bot.config import get_config
from bot.worker import process_file_task
from bot.utils import BotLogger

cfg = get_config()
MAX_UPLOAD_SIZE = cfg.MAX_UPLOAD_SIZE*1024*1024

class SizeLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        content_length = int(request.headers.get("content-length", 0))
        if content_length > MAX_UPLOAD_SIZE:
            return JSONResponse(
                status_code=413,
                content={"detail": "文件大小超过限制"}
            )
        return await call_next(request)

app = FastAPI(
    title="FakeVoi API",
    description="FakeVoi的API服务",
    version="0.0.1",
    middleware=[Middleware(SizeLimitMiddleware)]
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 文件存储配置
UPLOAD_PATH = cfg.UPLOAD_PATH
os.makedirs(UPLOAD_PATH, exist_ok=True)
ALLOWED_EXTENSIONS = {'wav'}

def allowed_file(filename: str) -> bool:
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.post("/upload",
         summary="上传语音文件",
         response_description="返回存储的文件信息",
         tags=["语音处理"])
async def upload_audio(file: UploadFile = File(..., description="音频文件，仅支持 WAV 格式")):
    try:
        # 验证文件类型
        if not allowed_file(file.filename):
            raise HTTPException(status_code=400, detail="不支持的文件格式")

        # 实际文件大小验证
        if file.size > MAX_UPLOAD_SIZE:
            raise HTTPException(413, "文件大小超过限制")

        # 生成安全文件名
        file_ext = file.filename.split('.')[-1]
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"{timestamp}.{file_ext}"
        save_path = os.path.join(UPLOAD_PATH, filename)

        # 保存文件
        with open(save_path, 'wb') as f:
            while chunk := await file.read(1024 * 1024):    # 1Mb chunks
                f.write(chunk)

        # 记录保存成功日志

        BotLogger.info(
            "文件保存成功",
            extra={
                "action": "file_saved",
                # 修改字段名称，添加前缀避免冲突
                "file_name": filename,          
                "file_size": file.size,         
                "content_type": file.content_type  
            }
        )

        # 发送异步任务
        task = process_file_task.delay(file_name=filename)
        # 确保任务对象有效
        if not isinstance(task, AsyncResult):
            BotLogger.error("Celery任务提交返回异常对象", extra={"file_name": filename})
            raise HTTPException(500, "Celery任务提交失败")

        # 记录任务提交日志
        BotLogger.info(
            "异步任务已提交",
            extra={
                "action": "task_submitted",
                "task_id": task.id,
                "file_name": filename
            }
        )
        
        return {
            "message": "文件上传成功，已进入处理队列",
            "file_name": filename,
            "task_id": task.id
        }

    except HTTPException as he:
        raise he
    
    except ConnectionError as ce:
        BotLogger.critical("消息队列连接失败", exc_info=True)
        raise HTTPException(503, "系统暂时不可用")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文件处理错误: {str(e)}")

# 任务状态查询接口
@app.get("/tasks/{task_id}",
         summary="获取任务执行结果",
         response_description="",
         tags=[""])
def get_task_status(task_id: str):
    task = AsyncResult(task_id)
    return {
        "task_id": task_id,
        "status": task.result.get("status") if task.ready() else "UNKNOWN",
        "path": task.result.get("path") if task.ready() else None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=cfg.MAIN_HOST,
        port=cfg.MAIN_PORT,
        reload=True,
        ssl_certfile=None,
        ssl_keyfile=None
    )