import unittest


import pipeline


class PipelineConfigTest(unittest.TestCase):
    def test_vad_uses_appropriate_silence_threshold(self):
        self.assertEqual(pipeline.VAD_MIN_SILENCE_DURATION_MS, 2000)


class PipelineResultJsonTest(unittest.TestCase):
    def test_run_returns_json_path_next_to_source_audio(self):
        """run()은 원본 오디오 옆 JSON 경로를 반환해야 한다."""
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            fake_audio = Path(f.name)

        try:
            with patch("pipeline.transcribe", return_value=[]), \
                 patch("pipeline.format_transcript", return_value=""), \
                 patch("pipeline.should_use_chunking", return_value=False):
                result_path = pipeline.run(str(fake_audio), title="저장 위치")

            self.assertEqual(result_path, str(fake_audio.with_suffix(".json").resolve()))
            Path(result_path).unlink(missing_ok=True)
        finally:
            fake_audio.unlink(missing_ok=True)

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

    def test_run_uses_original_audio_by_default(self):
        """기본 실행은 원본 오디오를 STT 입력으로 사용해야 한다."""
        import json
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            fake_audio = f.name

        try:
            captured = {}
            with patch("pipeline.run_vad") as mock_vad, \
                 patch("pipeline.transcribe") as mock_transcribe, \
                 patch("pipeline.format_transcript", return_value=""):
                def fake_transcribe(audio_path, **kwargs):
                    captured["audio_path"] = audio_path
                    return []

                mock_transcribe.side_effect = fake_transcribe

                result_path = pipeline.run(fake_audio, title="기본 경로")

            result = json.loads(Path(result_path).read_text())
            self.assertEqual(captured["audio_path"], str(Path(fake_audio).resolve()))
            mock_vad.assert_not_called()
            self.assertFalse(result["vad_applied"])
            self.assertEqual(result["source_audio_path"], str(Path(fake_audio).resolve()))
            self.assertEqual(result["stt_audio_path"], str(Path(fake_audio).resolve()))
            Path(result_path).unlink(missing_ok=True)
            Path(fake_audio).with_suffix(".json").unlink(missing_ok=True)
        finally:
            Path(fake_audio).unlink(missing_ok=True)

    def test_result_json_marks_chunking_when_long_audio_uses_chunked_path(self):
        """chunked STT 경로가 예상되면 JSON에 반영되어야 한다."""
        import json
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            fake_audio = f.name

        try:
            with patch("pipeline.should_use_chunking", return_value=True), \
                 patch("pipeline.transcribe", return_value=[]), \
                 patch("pipeline.format_transcript", return_value=""):
                result_path = pipeline.run(fake_audio, title="청크 메타데이터")

            result = json.loads(Path(result_path).read_text())
            self.assertTrue(result["chunking_applied"])
            Path(result_path).unlink(missing_ok=True)
            Path(fake_audio).with_suffix(".json").unlink(missing_ok=True)
        finally:
            Path(fake_audio).unlink(missing_ok=True)

    def test_pipeline_continues_when_doc_loading_fails(self):
        """문서 로드 실패 시에도 기본 STT는 계속되어야 한다."""
        import json
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            fake_audio = f.name

        try:
            with patch("pipeline.load_context", side_effect=RuntimeError("doc load failed")), \
                 patch("pipeline.transcribe", return_value=[]), \
                 patch("pipeline.format_transcript", return_value=""):
                result_path = pipeline.run(
                    fake_audio,
                    title="문서 실패",
                    doc_paths=["/tmp/missing.txt"],
                )

            result = json.loads(Path(result_path).read_text())
            self.assertFalse(result["term_metadata_applied"])
            self.assertEqual(result["doc_content"], "")
            Path(result_path).unlink(missing_ok=True)
            Path(fake_audio).with_suffix(".json").unlink(missing_ok=True)
        finally:
            Path(fake_audio).unlink(missing_ok=True)

    def test_result_json_marks_audio_enhanced(self):
        """오디오 보정이 적용되면 JSON에 기록되어야 한다."""
        import json
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            fake_audio = f.name

        try:
            enhanced_path = Path("/tmp/fake_enhanced.wav")
            with patch("pipeline.enhance_audio_for_stt", return_value=enhanced_path), \
                 patch("pipeline.transcribe", return_value=[]), \
                 patch("pipeline.format_transcript", return_value=""), \
                 patch("pipeline.should_use_chunking", return_value=False):
                result_path = pipeline.run(fake_audio, title="오디오 보정")

            result = json.loads(Path(result_path).read_text())
            self.assertTrue(result["audio_enhanced"])
            self.assertEqual(result["stt_audio_path"], str(enhanced_path))
            Path(result_path).unlink(missing_ok=True)
            Path(fake_audio).with_suffix(".json").unlink(missing_ok=True)
        finally:
            Path(fake_audio).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
