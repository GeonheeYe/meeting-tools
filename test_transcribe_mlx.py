import unittest
from unittest.mock import MagicMock, patch

import transcribe


class AppleSiliconDetectionTest(unittest.TestCase):
    def test_detects_apple_silicon_arm64(self):
        with patch("platform.machine", return_value="arm64"):
            self.assertTrue(transcribe._is_apple_silicon())

    def test_not_apple_silicon_x86(self):
        with patch("platform.machine", return_value="x86_64"):
            with patch("platform.processor", return_value="Intel"):
                self.assertFalse(transcribe._is_apple_silicon())


class RunWhisperDispatchTest(unittest.TestCase):
    def setUp(self):
        # 모델 캐시 초기화
        transcribe._WHISPER_MODEL = None

    def test_uses_mlx_on_apple_silicon(self):
        fake_segments = [{"start": 0.0, "end": 1.5, "text": "안녕하세요"}]
        with patch("transcribe._is_apple_silicon", return_value=True):
            with patch("transcribe._run_whisper_mlx", return_value=fake_segments) as mlx_fn:
                result = transcribe.run_whisper("audio.wav")
        mlx_fn.assert_called_once_with("audio.wav", None)
        self.assertEqual(result, fake_segments)

    def test_uses_faster_whisper_on_non_apple(self):
        fake_segments = [{"start": 0.0, "end": 1.5, "text": "안녕하세요"}]
        with patch("transcribe._is_apple_silicon", return_value=False):
            with patch("transcribe._run_whisper_faster", return_value=fake_segments) as fw_fn:
                result = transcribe.run_whisper("audio.wav", "컨텍스트")
        fw_fn.assert_called_once_with("audio.wav", "컨텍스트")
        self.assertEqual(result, fake_segments)

    def test_mlx_output_has_required_fields(self):
        """mlx-whisper 출력이 faster-whisper와 동일한 필드를 갖는지 확인."""
        import sys
        mock_mlx_result = {
            "segments": [
                {"start": 0.0, "end": 2.0, "text": " 테스트입니다", "id": 0}
            ]
        }
        # mlx_whisper가 설치되지 않은 환경에서도 테스트 가능하도록 가짜 모듈 주입
        fake_mlx_whisper = MagicMock()
        fake_mlx_whisper.transcribe.return_value = mock_mlx_result
        with patch.dict(sys.modules, {"mlx_whisper": fake_mlx_whisper}):
            result = transcribe._run_whisper_mlx("audio.wav", None)
        self.assertEqual(len(result), 1)
        self.assertIn("start", result[0])
        self.assertIn("end", result[0])
        self.assertIn("text", result[0])
        self.assertEqual(result[0]["text"], "테스트입니다")  # strip() 확인


if __name__ == "__main__":
    unittest.main()
