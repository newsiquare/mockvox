import os
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.requests import Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from celery.result import AsyncResult
import json
from pathlib import Path

from bot.config import get_config, UPLOAD_PATH, DENOISED_ROOT_PATH, SLICED_ROOT_PATH, ASR_PATH
from bot.worker import celeryApp, process_file_task, train_task
from bot.utils import BotLogger, generate_unique_filename

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
os.makedirs(UPLOAD_PATH, exist_ok=True)
ALLOWED_EXTENSIONS = {'wav'}

def allowed_file(filename: str) -> bool:
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.post(
    "/revision",
    summary="ASR校对",
    response_description="保存成功标志",
    tags=["ASR结果校对"]
)
async def asr_revision(
    filename: str = Form(..., description="文件名（调用 /upload 上传后返回的文件名, 无后缀名"),
    results: str = Form("{}", description="JSON格式的校对结果"),
    denoised: bool = Form(True, description="是否已降噪")
):
    results_list = json.loads(results)
    wav_root = DENOISED_ROOT_PATH if denoised else SLICED_ROOT_PATH
    wav_root = Path(wav_root) / filename
    asr_path = Path(ASR_PATH) / filename / 'output.json'

    try:
        if not isinstance(results_list, list):
            raise HTTPException(
                status_code=422,
                detail="Invalid format: results should be a JSON array"
            )

        for item in results_list:
            if not isinstance(item, dict):
                raise HTTPException(
                    status_code=422,
                    detail=f"Invalid item format: expected dict, got {type(item)}"
                )
                
            if "key" not in item or "text" not in item:
                raise HTTPException(
                    status_code=422,
                    detail="Invalid item: missing 'key' or 'text' field"
                )

            wav_path = wav_root / f"{item['key']}.wav"
            if not wav_path.exists():
                raise HTTPException(
                    status_code=404,
                    detail=f"Audio file not found: {item['key']}.wav"
                )

        with open(asr_path, "w", encoding="utf-8") as f:
            json.dump(results_list, f, ensure_ascii=False, indent=2)
            
        return {"success": True, "message": "ASR revision saved successfully"}
        
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=400,
            detail="Invalid JSON format in results parameter"
        )

@app.post(
    "/train",
    summary="启动训练",
    response_description="返回任务ID",
    tags=["模型训练"]
)
async def start_train(
    filename: str = Form(..., description="训练文件名（调用 /upload 上传后返回的文件名, 无后缀名"),
    epochs_sovits: int = Form(10, description="SoVITs训练轮次"),
    epochs_gpt: int = Form(10, description="GPT训练轮次"),
    denoised: bool = Form(True, description="是否已降噪"),
    config: str = Form("{}", description="JSON 格式的配置参数")
):
    try:
        # 发送异步任务
        task = train_task.delay(
            file_name=filename,
            sovits_epochs=epochs_sovits,
            gpt_epochs=epochs_gpt, 
            ifDenoise=denoised
        )
        # 确保任务对象有效
        if not isinstance(task, AsyncResult):
            BotLogger.error(f"Celery训练任务提交异常 | {filename}")
            raise HTTPException(500, "Celery训练任务提交失败")

        # 记录任务提交日志
        BotLogger.info(
            "训练任务已提交",
            extra={
                "action": "stage2_task_submitted",
                "task_id": task.id,
                "file_name": filename
            }
        )

        return {
            "message": "训练任务已进入处理队列",
            "file_name": filename,
            "task_id": task.id
        }

    except HTTPException as he:
        raise he
    
    except ConnectionError as ce:
        BotLogger.critical("消息队列连接失败", exc_info=True)
        raise HTTPException(503, "系统暂时不可用")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"训练过程错误: {str(e)}")


@app.post(
    "/upload",
    summary="上传语音文件",
    response_description="返回存储的文件信息和任务ID",
    tags=["语音处理"]
)
async def upload_audio(file: UploadFile = File(..., description="音频文件，仅支持 WAV 格式")):
    try:
        # 验证文件类型
        if not allowed_file(file.filename):
            raise HTTPException(status_code=400, detail="不支持的文件格式")

        # 实际文件大小验证
        if file.size > MAX_UPLOAD_SIZE:
            raise HTTPException(413, "文件大小超过限制")

        # 生成唯一文件名
        filename = generate_unique_filename(file.filename)
        save_path = os.path.join(UPLOAD_PATH, filename)

        # 保存文件
        with open(save_path, 'wb') as f:
            while chunk := await file.read(1024 * 1024):    # 1Mb chunks
                f.write(chunk)

        # 记录保存成功日志

        BotLogger.info(
            "文件已保存",
            extra={
                "action": "file_saved",
                "file_name": filename,          
                "file_size": file.size,         
                "content_type": file.content_type  
            }
        )

        # 发送异步任务
        task = process_file_task.delay(file_name=filename, ifDenoise=True)
        # 确保任务对象有效
        if not isinstance(task, AsyncResult):
            BotLogger.error(f"Celery文件任务提交异常 | {filename}")
            raise HTTPException(500, "Celery文件任务提交失败")

        # 记录任务提交日志
        BotLogger.info(
            "文件任务已提交",
            extra={
                "action": "stage1_task_submitted",
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
         response_description="返回任务状态",
         tags=[""])
def get_task_status(task_id: str):
    task = celeryApp.AsyncResult(task_id)
    return {
        "task_id": task_id,
        "status": task.result.get("status") if task.ready() else "UNKNOWN",
        "results": task.result.get("results") if task.ready() else None,
        "time": task.result.get("time") if task.ready() else None
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