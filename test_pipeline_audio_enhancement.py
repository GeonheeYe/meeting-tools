import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pipeline


class AudioEnhancementTest(unittest.TestCase):
    def test_audio_enhancement_filter_boosts_quiet_speech_more_aggressively(self):
        self.assertIn("loudnorm", pipeline.ENHANCED_AUDIO_FILTER)
        self.assertIn("dynaudnorm", pipeline.ENHANCED_AUDIO_FILTER)
        self.assertIn("acompressor", pipeline.ENHANCED_AUDIO_FILTER)

    def test_enhance_audio_for_stt_creates_file_next_to_source(self):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            source = Path(f.name)

        try:
            with patch("pipeline.subprocess.run") as run_fn:
                result = pipeline.enhance_audio_for_stt(source)

            self.assertEqual(result.parent, source.parent)
            self.assertEqual(result.name, f"{source.stem}_enhanced.wav")
            self.assertEqual(result.suffix, ".wav")
            run_fn.assert_called_once()
            command = run_fn.call_args.args[0]
            self.assertIn("-af", command)
            self.assertIn("dynaudnorm", " ".join(command))
            source.with_name(f"{source.stem}_enhanced.wav").unlink(missing_ok=True)
        finally:
            source.unlink(missing_ok=True)

    def test_enhance_audio_for_stt_reuses_stable_path_for_same_source(self):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            source = Path(f.name)

        try:
            with patch("pipeline.subprocess.run"):
                first = pipeline.enhance_audio_for_stt(source)
                second = pipeline.enhance_audio_for_stt(source)

            self.assertEqual(first, second)
            self.assertEqual(first, source.with_name(f"{source.stem}_enhanced.wav"))
            source.with_name(f"{source.stem}_enhanced.wav").unlink(missing_ok=True)
        finally:
            source.unlink(missing_ok=True)

    def test_pipeline_falls_back_when_audio_enhancement_fails(self):
        import json

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            source = Path(f.name)

        try:
            with patch("pipeline.enhance_audio_for_stt", side_effect=RuntimeError("enhance failed")), \
                 patch("pipeline.transcribe", return_value=[]), \
                 patch("pipeline.format_transcript", return_value=""), \
                 patch("pipeline.should_use_chunking", return_value=False):
                result_path = pipeline.run(str(source), title="enhance fallback")

            result = json.loads(Path(result_path).read_text())
            self.assertFalse(result["audio_enhanced"])
            self.assertEqual(result["stt_audio_path"], str(source.resolve()))
            Path(result_path).unlink(missing_ok=True)
            source.with_suffix(".json").unlink(missing_ok=True)
        finally:
            source.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
