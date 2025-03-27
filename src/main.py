# main.py
import os
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware import Middleware
from starlette.middleware.contentsecuritypolicy import ContentSecurityPolicyMiddleware
from .config import get_settings

settings = get_settings()
MAX_UPLOAD_SIZE = settings.MAX_UPLOAD_SIZE*1024*1024

app = FastAPI(
    title="语音克隆API",
    description="接收并存储语音样本的API服务",
    version="0.0.1",
    middleware=[Middleware(ContentSecurityPolicyMiddleware, max_upload_size=MAX_UPLOAD_SIZE)]
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
UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
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
        filename = f"recording_{timestamp}.{file_ext}"
        save_path = os.path.join(UPLOAD_DIR, filename)

        # 保存文件
        contents = await file.read()
        with open(save_path, 'wb') as f:
            f.write(contents)

        return {
            "message": "上传成功",
            "filename": filename,
            "path": save_path,
            "duration_ms": len(contents) // 44  # 估算时长（44 bytes/ms）
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文件处理错误: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=5000,
        reload=True,
        ssl_certfile=None,
        ssl_keyfile=None
    )