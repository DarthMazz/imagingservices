from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import logging
import json
import aiofiles
import easyocr
import numpy as np
from PIL import Image

app = FastAPI()

# ロガーの設定
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ディレクトリ設定
UPLOAD_DIR = "static/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# 静的ファイル（画像）とテンプレートの設定
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# OCRリーダーの初期化（日本語と英語）
reader = easyocr.Reader(["ja", "en"])


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload", response_class=HTMLResponse)
async def upload_image(request: Request, file: UploadFile = File(...)):
    # ファイル保存
    logger.info(f"Starting file upload: {file.filename}")
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    async with aiofiles.open(file_path, "wb") as out_file:
        while content := await file.read(1024 * 1024):  # 1MBずつ読み書き
            await out_file.write(content)
    logger.info(f"File saved to {file_path}")

    # OCR実行
    logger.info("Starting OCR processing")
    image = Image.open(file_path)
    image_np = np.array(image)
    results = reader.readtext(image_np)
    logger.info(f"OCR completed. Found {len(results)} text regions")
    if len(results) > 0:
        logger.info(f"First result: {results[0]}")

    # 結果をJSONシリアライズ可能な形式に変換
    ocr_data = []
    for bbox, text, prob in results:
        # bboxはnumpy型を含むため、標準のint型リストに変換
        points = [[int(p[0]), int(p[1])] for p in bbox]
        ocr_data.append({"points": points, "text": text})

    # アップロードした画像を表示するためのHTML断片を返す（HTMX用）
    image_url = f"/{file_path}"
    ocr_json = json.dumps(ocr_data)

    return f"""
    <div id="image-container" style="position: relative; display: inline-block;">
        <img src="{image_url}" id="target-image" style="max-width: 500px;" data-ocr='{ocr_json}'>
    </div>
    """
