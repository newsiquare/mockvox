import subprocess
import os
def clone_repository(target_dir, repo_url):
    if not os.path.exists(target_dir):

        parent_dir = os.path.dirname(target_dir)
        if parent_dir and not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)
        
        try:
            subprocess.run(
                ['git', 'clone', repo_url, target_dir],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT
            )
        except subprocess.CalledProcessError as e:
            raise
    else:
        pass

# 检查有没有初始模型
clone_repository("./pretrained/G2PWModel", "https://huggingface.co/alextomcat/G2PWModel.git")
clone_repository("./pretrained/GPT-SoVITS", "https://huggingface.co/lj1995/GPT-SoVITS.git")
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse,FileResponse
from fastapi.requests import Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from celery.result import AsyncResult
import json
from pathlib import Path
import glob
import torch

from mockvox.config import (
    get_config,
    SLICED_ROOT_PATH,
    DENOISED_ROOT_PATH,
    ASR_PATH,
    GPT_HALF_WEIGHTS_FILE,
    SOVITS_HALF_WEIGHTS_FILE,
    REF_AUDIO_PATH,
    OUT_PUT_PATH,
    UPLOAD_PATH,
    WEIGHTS_PATH,
    GPT_WEIGHTS_FILE,
    SOVITS_G_WEIGHTS_FILE 
)
from mockvox.worker import celeryApp, process_file_task, train_task, inference_task, resume_task
from mockvox.utils import MockVoxLogger, generate_unique_filename, allowed_file, i18n

cfg = get_config()

class SizeLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        content_length = int(request.headers.get("content-length", 0))
        if content_length > cfg.MAX_UPLOAD_SIZE:
            return JSONResponse(
                status_code=413,
                content={"detail": i18n("文件大小超过限制")}
            )
        return await call_next(request)

app = FastAPI(
    title="MockVox API",
    description=i18n("MockVox的API服务"),
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

@app.post(
    "/revision",
    summary=i18n("ASR校对"),
    response_description=i18n("保存成功"),
    tags=[i18n("ASR结果校对")]
)
async def asr_revision(
    file_id: str = Form(..., description=i18n("文件ID (调用 /upload 上传后返回的文件ID)")),
    results: str = Form("{}", description=i18n("JSON格式的校对结果")),
    denoised: bool = Form(True, description=i18n("是否已降噪"))
):
    results_list = json.loads(results)
    wav_root = DENOISED_ROOT_PATH if denoised else SLICED_ROOT_PATH
    wav_root = Path(wav_root) / file_id
    asr_path = Path(ASR_PATH) / file_id / 'output.json'

    with open(asr_path, "r", encoding="utf-8") as f:
        output_data = json.load(f)

    try:
        if not isinstance(results_list, list):
            raise HTTPException(
                status_code=422,
                detail=i18n("无效格式: 结果应为JSON数组")
            )

        for item in results_list:
            if not isinstance(item, dict):
                raise HTTPException(
                    status_code=422,
                    detail=f"{i18n('无效项格式: 应为字典(dict)，实际为')} {type(item)}"
                )
                
            if "key" not in item or "text" not in item:
                raise HTTPException(
                    status_code=422,
                    detail=i18n("无效项：缺少 key 或 text 字段")
                )

            wav_path = wav_root / f"{item['key']}.wav"
            if not wav_path.exists():
                raise HTTPException(
                    status_code=404,
                    detail=f"{i18n('音频文件未找到:')} {item['key']}.wav"
                )

        output_data["results"] = results_list
        with open(asr_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
            
        return {"success": True, "message": i18n("ASR校对保存成功")}
        
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=400,
            detail=i18n("results参数中的JSON格式无效")
        )

@app.post(
    "/train",
    summary=i18n("启动训练"),
    response_description=i18n("返回任务ID"),
    tags=[i18n("模型训练")]
)
async def start_train(
    file_id: str = Form(..., description=i18n("文件ID (调用 /upload 上传后返回的文件ID)")),
    epochs_sovits: int = Form(1, description=i18n("SoVITs訓練輪次")),
    epochs_gpt: int = Form(1, description=i18n("GPT训练轮次")),
    version: str = Form('v4', description=i18n("版本")),
    denoised: bool = Form(True, description=i18n("是否已降噪")),
    config: str = Form("{}", description=i18n("JSON 格式的配置参数"))
):
    try:
        # 发送异步任务
        task = train_task.delay(
            file_id=file_id,
            sovits_epochs=epochs_sovits,
            gpt_epochs=epochs_gpt, 
            version=version,
            ifDenoise=denoised
        )
        # 确保任务对象有效
        if not isinstance(task, AsyncResult):
            MockVoxLogger.error(f"{i18n('Celery训练任务提交失败')} | {file_id}")
            raise HTTPException(500, i18n("Celery训练任务提交失败"))

        # 记录任务提交日志
        MockVoxLogger.info(
            i18n("训练任务已进入Celery处理队列"),
            extra={
                "action": "stage2_task_submitted",
                "task_id": task.id,
                "file_id": file_id
            }
        )

        return {
            "message": i18n("训练任务已进入Celery处理队列"),
            "task_id": task.id
        }

    except HTTPException as he:
        raise he
    
    except ConnectionError as ce:
        MockVoxLogger.critical(i18n("消息队列连接失败"), exc_info=True)
        raise HTTPException(503, i18n("系统暂时不可用"))

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{i18n('训练过程错误')}: {str(e)}")

@app.post(
    "/resume",
    summary=i18n("继续训练"),
    response_description=i18n("返回任务ID"),
    tags=[i18n("模型训练")]
)
async def resume(
    model_id: str = Form(..., description=i18n("模型ID (调用 /train 返回的模型ID)")),
    epochs_sovits: int = Form(2, description=i18n("SoVITs訓練輪次")),
    epochs_gpt: int = Form(2, description=i18n("GPT训练轮次")),
    config: str = Form("{}", description=i18n("JSON 格式的配置参数"))
):
    try:
        # 发送异步任务
        task = resume_task.delay(
            model_id=model_id,
            sovits_epochs=epochs_sovits,
            gpt_epochs=epochs_gpt
        )
        # 确保任务对象有效
        if not isinstance(task, AsyncResult):
            MockVoxLogger.error(f"{i18n('Celery训练任务提交失败')} | {model_id}")
            raise HTTPException(500, i18n("Celery训练任务提交失败"))

        # 记录任务提交日志
        MockVoxLogger.info(
            f"{i18n('训练任务已进入Celery处理队列')} \n"
            f"task_id: {task.id} \n"
            f"model_id: {model_id}"
        )

        return {
            "message": i18n("训练任务已进入Celery处理队列"),
            "task_id": task.id
        }

    except HTTPException as he:
        raise he
    
    except ConnectionError as ce:
        MockVoxLogger.critical(i18n("消息队列连接失败"), exc_info=True)
        raise HTTPException(503, i18n("系统暂时不可用"))

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{i18n('训练过程错误')}: {str(e)}")

@app.post(
    "/inference",
    summary=i18n("启动推理"),
    response_description=i18n("返回任务ID"),
    tags=[i18n("模型推理")]
)
async def start_inference(
    model_id:str = Form(..., description=i18n("模型id")), 
    ref_audio_file_id:str = Form(..., description=i18n("参考音频文件ID (调用 /uploadRef 上传后返回的参考音频文件ID)")),
    ref_text:str = Form(..., description=i18n("参考音频的文字")), 
    ref_language:str = Form('zh', description=i18n("参考音频的语言")), 
    target_text:str = Form(..., description=i18n("生成音频的文字")), 
    target_language:str = Form('zh', description=i18n("生成音频的语言")), 
    top_p:float = Form(1, description=i18n("top_p")), 
    top_k:int = Form(15, description=i18n("GPT采样参数(无参考文本时不要太低。不懂就用默认)")), 
    temperature:float = Form(1, description=i18n("temperature")), 
    speed:float = Form(1, description=i18n("语速")),
):
    try:
        gpt_path = Path(WEIGHTS_PATH) / model_id / GPT_HALF_WEIGHTS_FILE
        if not gpt_path.exists():
            MockVoxLogger.error(i18n("路径错误! 找不到GPT模型"))
            return
        gpt_ckpt = torch.load(gpt_path, map_location="cpu")
        version = gpt_ckpt["config"]["model"]["version"]
        MockVoxLogger.info(f"Model Version: {version}")
        sovits_path = Path(WEIGHTS_PATH) / model_id / SOVITS_HALF_WEIGHTS_FILE
        if not sovits_path.exists():
            MockVoxLogger.error(i18n("路径错误! 找不到SOVITS模型"))
            return
        filename = ''
        for file in glob.glob(os.path.join(Path(REF_AUDIO_PATH),ref_audio_file_id+".*")):
            filename = file
        if filename == '':
            MockVoxLogger.error(i18n("请上传参考音频"))
            return
        # 发送异步任务
        task = inference_task.delay(
            gpt_model_path=str(gpt_path),                   
            soVITS_model_path=str(sovits_path) , 
            ref_audio_path=str(Path(REF_AUDIO_PATH)/filename), 
            ref_text=ref_text , 
            ref_language=ref_language, 
            target_text=target_text , 
            target_language=target_language ,
            top_p=top_p , 
            top_k=top_k , 
            temperature=temperature , 
            speed=speed,
            version=version
        )
        # 确保任务对象有效
        if not isinstance(task, AsyncResult):
            MockVoxLogger.error(i18n("Celery推理任务提交失败"))
            raise HTTPException(500, i18n("Celery推理任务提交失败"))

        MockVoxLogger.info(
            f"{i18n('推理任务已进入Celery处理队列')} \n"
            f"task_id: {task.id} \n"
            f"model_id: {model_id}"
        )
        return {
            "message": i18n("推理任务已进入Celery处理队列"),
            "task_id": task.id
        }

    except HTTPException as he:
        raise he
    
    except ConnectionError as ce:
        MockVoxLogger.critical(i18n("消息队列连接失败"), exc_info=True)
        raise HTTPException(503, i18n("系统暂时不可用"))

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{i18n('推理过程错误')}: {str(e)}")

@app.get(
    "/output/{task_id}",
    summary=i18n("下载推理结果文件"),
    response_description=i18n("下载推理结果文件"),
    tags=[i18n("下载推理结果文件")]
)
async def download_outputs(task_id:str):
    filename = task_id + ".WAV"
    if not os.path.exists(os.path.join(OUT_PUT_PATH,filename)):
        MockVoxLogger.error(i18n("音频文件不存在"))
        return

    return FileResponse(os.path.join(OUT_PUT_PATH,filename))

@app.post(
    "/uploadRef",
    summary=i18n("请上传参考音频"),
    response_description=i18n("返回文件ID"),
    tags=[i18n("请上传参考音频")]
)
async def upload_ref_audio(
    file: UploadFile = File(..., description=i18n("请上传3~10秒内参考音频，超过会报错！")+i18n("音频文件支持 .WAV .MP3 .FLAC 格式"))
):
    # 验证文件类型
    if not allowed_file(file.filename):
        raise HTTPException(status_code=400, detail=i18n("不支持的文件格式"))

    # 实际文件大小验证
    if file.size > cfg.MAX_UPLOAD_SIZE:
        raise HTTPException(413, i18n("文件大小超过限制"))

    # 生成唯一文件名
    filename = generate_unique_filename(file.filename)
    if not os.path.exists(REF_AUDIO_PATH):
        os.makedirs(REF_AUDIO_PATH, exist_ok=True)
    save_path = os.path.join(REF_AUDIO_PATH, filename)

    # 保存文件
    with open(save_path, 'wb') as f:
        while chunk := await file.read(1024 * 1024):    # 1Mb chunks
            f.write(chunk)
    return {"file_id": Path(filename).stem}

@app.post(
    "/upload",
    summary=i18n("上传语音文件"),
    response_description=i18n("返回存储的文件信息和任务ID"),
    tags=[i18n("语音文件切片及降噪")]
)
async def upload_audio(
    file: UploadFile = File(..., description=i18n("音频文件支持 .WAV .MP3 .FLAC 格式")),
    language: str = Form('zh', description=i18n("语言"))
):
    try:
        # 验证文件类型
        if not allowed_file(file.filename):
            raise HTTPException(status_code=400, detail=i18n("不支持的文件格式"))

        # 实际文件大小验证
        if file.size > cfg.MAX_UPLOAD_SIZE:
            raise HTTPException(413, i18n("文件大小超过限制"))

        # 生成唯一文件名
        filename = generate_unique_filename(file.filename)
        save_path = os.path.join(UPLOAD_PATH, filename)

        # 保存文件
        with open(save_path, 'wb') as f:
            while chunk := await file.read(1024 * 1024):    # 1Mb chunks
                f.write(chunk)

        # 记录保存成功日志

        MockVoxLogger.info(
            i18n("保存成功"),
            extra={
                "action": "file_saved",
                "file_id": Path(filename).stem,          
                "file_size": file.size,         
                "content_type": file.content_type  
            }
        )

        # 发送异步任务
        task = process_file_task.delay(file_name=filename, language=language, ifDenoise=True)
        # 确保任务对象有效
        if not isinstance(task, AsyncResult):
            MockVoxLogger.error(f"{i18n('Celery文件处理任务提交失败')} | {filename}")
            raise HTTPException(500, i18n("Celery文件处理任务提交失败"))

        # 记录任务提交日志
        MockVoxLogger.info(
            i18n("文件上传成功, 已进入Celery处理队列"),
            extra={
                "action": "stage1_task_submitted",
                "task_id": task.id,
                "file_id": Path(filename).stem
            }
        )
        
        return {
            "message": i18n("文件上传成功, 已进入Celery处理队列"),
            "file_id": Path(filename).stem,
            "task_id": task.id
        }

    except HTTPException as he:
        raise he
    
    except ConnectionError as ce:
        MockVoxLogger.critical(i18n("消息队列连接失败"), exc_info=True)
        raise HTTPException(503, i18n("系统暂时不可用"))

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{i18n('文件处理错误')}: {str(e)}")

# 任务状态查询接口
@app.get("/tasks/{task_id}",
         summary=i18n("获取任务状态及执行结果"),
         response_description=i18n("返回任务状态及执行结果"),
         tags=[i18n("获取任务状态及执行结果")])
def get_task_status(task_id: str):
    task = celeryApp.AsyncResult(task_id)
    return {
        "task_id": task_id,
        "status": task.result.get("status") if task.ready() else "UNKNOWN",
        "results": task.result.get("results") if task.ready() else None,
        "time": task.result.get("time") if task.ready() else None
    }

# 模型信息查询接口
@app.get("/model/{model_id}",
         summary=i18n("获取模型信息"),
         response_description=i18n("返回模型版本及训练轮次"),
         tags=[i18n("获取模型信息")])
def get_model_info(model_id: str):
    gpt_weights = Path(WEIGHTS_PATH) / model_id / GPT_WEIGHTS_FILE
    sovits_weights = Path(WEIGHTS_PATH) / model_id / SOVITS_G_WEIGHTS_FILE
    if not gpt_weights.exists():
        return i18n("路径错误! 找不到GPT模型")
    if not sovits_weights.exists():
        return i18n("路径错误! 找不到SOVITS模型")

    gpt_ckpt = torch.load(gpt_weights, map_location="cpu")
    version = gpt_ckpt["config"]["model"]["version"]
    
    sovits_ckpt = torch.load(sovits_weights, map_location="cpu")
    
    gpt_epoch = gpt_ckpt["iteration"]
    sovits_epoch = sovits_ckpt["iteration"]

    return {
        "Model Version": version,
        "SoVITS trained epoch": sovits_epoch,
        "GPT trained epoch": gpt_epoch
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