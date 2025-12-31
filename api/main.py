from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import aiofiles

app = FastAPI()

# ディレクトリ設定
UPLOAD_DIR = "static/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# 静的ファイル（画像）とテンプレートの設定
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload", response_class=HTMLResponse)
async def upload_image(request: Request, file: UploadFile = File(...)):
    # ファイル保存
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    async with aiofiles.open(file_path, "wb") as out_file:
        while content := await file.read(1024 * 1024):  # 1MBずつ読み書き
            await out_file.write(content)

    # アップロードした画像を表示するためのHTML断片を返す（HTMX用）
    image_url = f"/{file_path}"
    return f"""
    <div id="image-container" style="position: relative; display: inline-block;">
        <img src="{image_url}" id="target-image" style="max-width: 500px;">
    </div>
    """
