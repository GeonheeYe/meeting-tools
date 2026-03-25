import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

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
             patch("pipeline.get_speech_timestamps", return_value=fake_timestamps) as timestamps_fn, \
             patch("pipeline.collect_chunks", return_value=fake_chunks), \
             patch("pipeline.save_audio"):
            load_fn.return_value = MagicMock()
            result = pipeline.run_vad(Path("/tmp/test.wav"))

        self.assertEqual(read_fn.call_count, 1)
        timestamps_fn.assert_called_once()
        self.assertEqual(timestamps_fn.call_args.kwargs["speech_pad_ms"], 1000)

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


class PipelineVadPolicyTest(unittest.TestCase):
    def test_run_always_passes_speech_timestamps_to_transcribe(self):
        """VAD는 항상 실행되고 speech_timestamps가 transcribe에 전달되어야 한다."""
        import json
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            fake_audio = f.name

        try:
            captured = {}
            fake_timestamps = [{"start": 0, "end": 16000}]
            with patch("pipeline.read_audio"), \
                 patch("pipeline.load_silero_vad"), \
                 patch("pipeline.get_speech_timestamps", return_value=fake_timestamps), \
                 patch("pipeline.transcribe") as mock_transcribe, \
                 patch("pipeline.format_transcript", return_value=""):
                def fake_transcribe(audio_path, **kwargs):
                    captured["speech_timestamps"] = kwargs.get("speech_timestamps")
                    return []

                mock_transcribe.side_effect = fake_transcribe

                result_path = pipeline.run(fake_audio, title="VAD 테스트")

            result = json.loads(Path(result_path).read_text())
            self.assertEqual(captured["speech_timestamps"], fake_timestamps)
            self.assertTrue(result["vad_applied"])
            Path(result_path).unlink(missing_ok=True)
            Path(fake_audio).with_suffix(".json").unlink(missing_ok=True)
        finally:
            Path(fake_audio).unlink(missing_ok=True)

    def test_run_falls_back_when_vad_fails(self):
        """VAD 실패 시 speech_timestamps=None으로 fallback해야 한다."""
        import json
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            fake_audio = f.name

        try:
            captured = {}
            with patch("pipeline.read_audio", side_effect=RuntimeError("vad failed")), \
                 patch("pipeline.transcribe") as mock_transcribe, \
                 patch("pipeline.format_transcript", return_value=""):
                def fake_transcribe(audio_path, **kwargs):
                    captured["speech_timestamps"] = kwargs.get("speech_timestamps")
                    return []

                mock_transcribe.side_effect = fake_transcribe

                result_path = pipeline.run(fake_audio, title="VAD fallback")

            result = json.loads(Path(result_path).read_text())
            self.assertIsNone(captured["speech_timestamps"])
            self.assertFalse(result["vad_applied"])
            Path(result_path).unlink(missing_ok=True)
            Path(fake_audio).with_suffix(".json").unlink(missing_ok=True)
        finally:
            Path(fake_audio).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
