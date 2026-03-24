import unittest


import pipeline


class PipelineConfigTest(unittest.TestCase):
    def test_vad_uses_3s_silence_threshold(self):
        self.assertEqual(pipeline.VAD_MIN_SILENCE_DURATION_MS, 3000)


class PipelineResultJsonTest(unittest.TestCase):
    def test_result_json_includes_agenda_items(self):
        """result JSON에 agenda_items 키가 포함되어야 한다."""
        import json
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        # 실제 오디오 파일 대신 빈 wav 파일을 임시 생성하여 FileNotFoundError 우회
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            fake_audio = f.name

        try:
            with patch("pipeline.run_vad") as mock_vad, \
                 patch("pipeline.transcribe") as mock_transcribe, \
                 patch("pipeline.format_transcript") as mock_fmt:
                mock_vad.return_value = Path("/tmp/fake_vad.wav")
                mock_transcribe.return_value = []
                mock_fmt.return_value = ""

                result_path = pipeline.run(fake_audio, title="테스트")

            result = json.loads(Path(result_path).read_text())
            self.assertIn("agenda_items", result)
            self.assertIsInstance(result["agenda_items"], list)
            self.assertEqual(result["agenda_items"], [])  # doc_paths 없으면 빈 리스트
            Path(result_path).unlink(missing_ok=True)
        finally:
            Path(fake_audio).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
