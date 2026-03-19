import unittest
from unittest.mock import patch

import transcribe


class WhisperModelCacheTest(unittest.TestCase):
    def test_get_whisper_model_reuses_single_instance(self):
        transcribe._WHISPER_MODEL = None

        with patch("transcribe.WhisperModel") as whisper_model_cls:
            whisper_model_cls.return_value = object()

            first = transcribe.get_whisper_model()
            second = transcribe.get_whisper_model()

        self.assertIs(first, second)
        self.assertEqual(whisper_model_cls.call_count, 1)


class PyannoteModelCacheTest(unittest.TestCase):
    def test_get_pyannote_pipeline_reuses_single_instance(self):
        transcribe._PYANNOTE_PIPELINE = None

        with patch("transcribe.Pipeline") as pipeline_cls:
            pipeline_cls.from_pretrained.return_value = object()

            first = transcribe.get_pyannote_pipeline("fake-token")
            second = transcribe.get_pyannote_pipeline("fake-token")

        self.assertIs(first, second)
        self.assertEqual(pipeline_cls.from_pretrained.call_count, 1)


if __name__ == "__main__":
    unittest.main()
