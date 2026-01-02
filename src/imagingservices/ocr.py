import logging

import easyocr
import numpy as np
from PIL import Image, UnidentifiedImageError


class OCRError(Exception):
    """OCR処理中に発生する独自の例外クラス"""

    # エラーコード定数（クライアントはこれを使ってエラー種別を判定可能）
    SYSTEM_ERROR = "SYSTEM_ERROR"
    IMAGE_ERROR = "IMAGE_ERROR"

    def __init__(self, message, code=SYSTEM_ERROR):
        self.code = code
        super().__init__(message)


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
        # 1. 入力パラメータの検証（パラメータ名を明示してValueErrorを出す）
        if not isinstance(image_path, str) or not image_path:
            raise ValueError(f"Invalid argument 'image_path': expected a non-empty string, got {type(image_path)}")

        try:
            image = Image.open(image_path)
            image_np = np.array(image)
            results = OCRService._reader.readtext(image_np)
            return results
        except FileNotFoundError:
            # ファイルパス間違いなどはクライアントの責任なので、標準例外をそのまま通す
            self._logger.warning(f"Image file not found: {image_path}")
            raise
        except (UnidentifiedImageError, Exception) as e:
            # 内部処理の失敗は独自例外にラップし、エラーコードを付与する
            code = OCRError.IMAGE_ERROR if isinstance(e, UnidentifiedImageError) else OCRError.SYSTEM_ERROR
            self._logger.error(f"OCR processing failed: {e}")
            raise OCRError(f"Failed to process image: {image_path}", code=code) from e

    def format_results(self, results):
        """OCR結果をJSONシリアライズ可能な形式（辞書のリスト）に変換します。"""
        ocr_data = []
        for bbox, text, prob in results:
            # bboxはnumpy型を含むため、標準のint型リストに変換
            points = [[int(p[0]), int(p[1])] for p in bbox]
            ocr_data.append({"points": points, "text": text})
        return ocr_data
