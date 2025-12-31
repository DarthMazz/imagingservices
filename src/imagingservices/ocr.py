import logging
import easyocr
import numpy as np
from PIL import Image


class OCRService:
    # OCRリーダーのインスタンスを保持するクラス変数
    _reader = None

    def __init__(self):
        """初期化時にOCRリーダーをロードします（シングルトンパターン）"""
        if OCRService._reader is None:
            # 日本語と英語を設定
            OCRService._reader = easyocr.Reader(["ja", "en"])
        self._logger = logging.getLogger(__name__)

    def process_image(self, image_path: str):
        """指定された画像のOCRを実行し、結果を返します。"""
        image = Image.open(image_path)
        image_np = np.array(image)
        results = OCRService._reader.readtext(image_np)
        return results

    def format_results(self, results):
        """OCR結果をJSONシリアライズ可能な形式（辞書のリスト）に変換します。"""
        ocr_data = []
        for bbox, text, prob in results:
            # bboxはnumpy型を含むため、標準のint型リストに変換
            points = [[int(p[0]), int(p[1])] for p in bbox]
            ocr_data.append({"points": points, "text": text})
        return ocr_data
