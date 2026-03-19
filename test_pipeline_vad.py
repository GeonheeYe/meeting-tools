import unittest
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pipeline


class VadSingleReadTest(unittest.TestCase):
    def test_read_audio_called_once(self):
        """run_vad() 실행 시 read_audio가 정확히 1번만 호출되는지 확인."""
        fake_wav = MagicMock()
        fake_wav.__len__ = lambda s: 16000 * 10  # 10초

        fake_timestamps = [{"start": 0, "end": 8000}, {"start": 16000, "end": 32000}]
        fake_chunks = MagicMock()
        fake_chunks.__len__ = lambda s: 16000 * 8  # 8초

        with patch("pipeline.load_silero_vad") as load_fn, \
             patch("pipeline.read_audio", return_value=fake_wav) as read_fn, \
             patch("pipeline.get_speech_timestamps", return_value=fake_timestamps), \
             patch("pipeline.collect_chunks", return_value=fake_chunks), \
             patch("pipeline.save_audio"):
            load_fn.return_value = MagicMock()
            result = pipeline.run_vad(Path("/tmp/test.wav"))

        self.assertEqual(read_fn.call_count, 1)

    def test_run_vad_returns_original_on_no_speech(self):
        """발화 구간이 없으면 원본 파일 경로를 반환하는지 확인."""
        fake_wav = MagicMock()
        fake_wav.__len__ = lambda s: 16000 * 5

        with patch("pipeline.load_silero_vad") as load_fn, \
             patch("pipeline.read_audio", return_value=fake_wav), \
             patch("pipeline.get_speech_timestamps", return_value=[]):
            load_fn.return_value = MagicMock()
            original = Path("/tmp/audio.wav")
            result = pipeline.run_vad(original)

        self.assertEqual(result, original)


if __name__ == "__main__":
    unittest.main()
