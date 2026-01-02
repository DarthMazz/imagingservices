import numpy as np
import pytest
from PIL import UnidentifiedImageError

from imagingservices import OCRError, OCRService


class TestOCRService:
    @pytest.fixture(autouse=True)
    def setup(self):
        # テスト間の干渉を防ぐため、シングルトンのインスタンス（クラス変数）をリセット
        OCRService._reader = None

    def test_singleton_initialization(self, mocker):
        """初期化がシングルトンパターンとして機能し、Readerが一度だけ生成されることを確認"""
        mock_reader_cls = mocker.patch("imagingservices.ocr.easyocr.Reader")
        # 1回目のインスタンス化
        service1 = OCRService()
        # 2回目のインスタンス化
        service2 = OCRService()

        # Readerの初期化は1回だけ呼ばれるはず
        mock_reader_cls.assert_called_once_with(["ja", "en"])

        # クラス変数がセットされていることを確認
        assert OCRService._reader is not None

    def test_process_image(self, mocker):
        """process_imageメソッドが画像を読み込み、OCRを実行することを確認"""
        # モックの設定
        mock_reader_cls = mocker.patch("imagingservices.ocr.easyocr.Reader")
        mock_image_open = mocker.patch("imagingservices.ocr.Image.open")
        mocker.patch("imagingservices.ocr.np.array")
        mock_reader_instance = mocker.MagicMock()
        mock_reader_cls.return_value = mock_reader_instance

        # ダミーのOCR結果
        expected_results = [([[10, 10], [50, 10], [50, 30], [10, 30]], "TEST", 0.99)]
        mock_reader_instance.readtext.return_value = expected_results

        # サービスの初期化と実行
        service = OCRService()
        results = service.process_image("dummy/path/image.jpg")

        # 検証
        mock_image_open.assert_called_with("dummy/path/image.jpg")
        mock_reader_instance.readtext.assert_called()
        assert results == expected_results

    def test_format_results(self, mocker):
        """format_resultsメソッドがnumpy型を標準型に変換し、正しい形式で返すことを確認"""
        # テストデータ: numpyの型を含むOCR結果
        bbox = [
            [np.int32(10), np.int32(10)],
            [np.int32(50), np.int32(10)],
            [np.int32(50), np.int32(30)],
            [np.int32(10), np.int32(30)],
        ]
        text = "TEST_TEXT"
        prob = np.float64(0.98)

        raw_results = [(bbox, text, prob)]

        # Readerの初期化をスキップするためにクラス変数をダミーで埋める
        OCRService._reader = mocker.MagicMock()
        service = OCRService()

        formatted_results = service.format_results(raw_results)

        # 期待される結果
        expected_points = [[10, 10], [50, 10], [50, 30], [10, 30]]

        assert len(formatted_results) == 1
        assert formatted_results[0]["text"] == "TEST_TEXT"
        assert formatted_results[0]["points"] == expected_points

        # 型の検証: numpy.int32 ではなく int になっているか
        assert isinstance(formatted_results[0]["points"][0][0], int)

    def test_process_image_invalid_input(self, mocker):
        """無効な入力パラメータでValueErrorが発生することを確認"""
        # Readerの初期化をスキップするためにクラス変数をダミーで埋める
        OCRService._reader = mocker.MagicMock()
        service = OCRService()

        # Noneの場合
        with pytest.raises(ValueError, match="Invalid argument 'image_path'"):
            service.process_image(None)

        # 空文字の場合
        with pytest.raises(ValueError, match="Invalid argument 'image_path'"):
            service.process_image("")

    def test_process_image_file_not_found(self, mocker):
        """ファイルが存在しない場合にFileNotFoundErrorがそのまま送出されることを確認"""
        OCRService._reader = mocker.MagicMock()
        mocker.patch("imagingservices.ocr.Image.open", side_effect=FileNotFoundError)

        service = OCRService()

        with pytest.raises(FileNotFoundError):
            service.process_image("non_existent.jpg")

    def test_process_image_ocr_error_image(self, mocker):
        """画像読み込みエラー時にOCRError(IMAGE_ERROR)が発生することを確認"""
        OCRService._reader = mocker.MagicMock()
        mocker.patch(
            "imagingservices.ocr.Image.open", side_effect=UnidentifiedImageError
        )

        service = OCRService()

        with pytest.raises(OCRError) as excinfo:
            service.process_image("corrupted.jpg")
        assert excinfo.value.code == OCRError.IMAGE_ERROR

    def test_process_image_ocr_error_system(self, mocker):
        """予期せぬエラー時にOCRError(SYSTEM_ERROR)が発生することを確認"""
        OCRService._reader = mocker.MagicMock()
        OCRService._reader.readtext.side_effect = Exception("Unexpected error")
        mocker.patch("imagingservices.ocr.Image.open")
        mocker.patch("imagingservices.ocr.np.array")

        service = OCRService()

        with pytest.raises(OCRError) as excinfo:
            service.process_image("valid.jpg")
        assert excinfo.value.code == OCRError.SYSTEM_ERROR
        assert excinfo.value.code == OCRError.SYSTEM_ERROR
