import unittest
from unittest.mock import MagicMock, call, patch

import transcribe


class ParallelTranscribeTest(unittest.TestCase):
    def test_whisper_and_diarization_both_called(self):
        """화자분리 있을 때 STT와 화자분리가 모두 호출되는지 확인."""
        fake_segments = [{"start": 0.0, "end": 1.0, "text": "안녕"}]
        fake_turns = [("SPEAKER_00", 0.0, 1.0)]

        with patch("transcribe.run_whisper", return_value=fake_segments) as stt_fn:
            with patch("transcribe.run_diarization", return_value=fake_turns) as diar_fn:
                result = transcribe.transcribe("audio.wav", num_speakers=2)

        stt_fn.assert_called_once()
        diar_fn.assert_called_once()

    def test_sequential_when_no_diarization(self):
        """화자분리 없을 때 run_diarization이 호출되지 않는지 확인."""
        fake_segments = [{"start": 0.0, "end": 1.0, "text": "안녕"}]

        with patch("transcribe.run_whisper", return_value=fake_segments):
            with patch("transcribe.run_diarization") as diar_fn:
                result = transcribe.transcribe("audio.wav", skip_diarization=True)

        diar_fn.assert_not_called()

    def test_parallel_result_merges_correctly(self):
        """병렬 실행 결과가 올바르게 병합되는지 확인."""
        fake_segments = [
            {"start": 0.0, "end": 1.0, "text": "첫 번째"},
            {"start": 1.5, "end": 2.5, "text": "두 번째"},
        ]
        fake_turns = [
            ("SPEAKER_00", 0.0, 1.0),
            ("SPEAKER_01", 1.5, 2.5),
        ]

        with patch("transcribe.run_whisper", return_value=fake_segments):
            with patch("transcribe.run_diarization", return_value=fake_turns):
                result = transcribe.transcribe("audio.wav", num_speakers=2)

        self.assertEqual(len(result), 2)
        speakers = {item["speaker"] for item in result}
        self.assertEqual(len(speakers), 2)  # 2명 화자


if __name__ == "__main__":
    unittest.main()
