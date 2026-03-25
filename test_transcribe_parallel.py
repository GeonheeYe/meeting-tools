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

    def test_skip_diarization_uses_chunked_whisper_path(self):
        """기본 STT 경로는 chunked whisper 경로를 사용해야 한다."""
        fake_segments = [{"start": 0.0, "end": 1.0, "text": "안녕"}]

        with patch("transcribe.run_whisper_chunked", return_value=fake_segments) as chunked_fn:
            with patch("transcribe.run_whisper") as whisper_fn:
                result = transcribe.transcribe("audio.wav", skip_diarization=True)

        chunked_fn.assert_called_once_with("audio.wav", initial_prompt=None, speech_timestamps=None)
        whisper_fn.assert_not_called()
        self.assertEqual(result[0]["speaker"], "Speaker A")


class PromptAndChunkMergeTest(unittest.TestCase):
    def test_chunking_uses_more_conservative_overlap(self):
        self.assertEqual(transcribe.DEFAULT_OVERLAP_SEC, 3.0)

    def test_extract_audio_chunk_uses_unique_temp_path_per_call(self):
        with patch("transcribe.subprocess.run"):
            first = transcribe._extract_audio_chunk("audio.wav", 0.0, 10.0, 0)
            second = transcribe._extract_audio_chunk("audio.wav", 10.0, 20.0, 0)

        self.assertNotEqual(first, second)

    def test_build_initial_prompt_includes_priority_terms_and_aliases(self):
        prompt = transcribe.build_initial_prompt(
            meeting_title="목표합의서 회의",
            context="RAG, VOC",
            term_metadata={
                "priority_terms": ["AEGIS-AP", "WiNG"],
                "alias_map": {"wing": "WiNG"},
            },
        )

        self.assertIn("회의 주제: 목표합의서 회의", prompt)
        self.assertIn("기본 컨텍스트: RAG, VOC", prompt)
        self.assertIn("핵심 용어: AEGIS-AP, WiNG", prompt)
        self.assertIn("wing -> WiNG", prompt)

    def test_merge_chunk_segments_removes_overlap_duplicates(self):
        chunks = [
            [
                {"start": 0.0, "end": 1.0, "text": "안녕하세요"},
                {"start": 1.0, "end": 2.0, "text": "회의를 시작하겠습니다"},
            ],
            [
                {"start": 1.8, "end": 2.8, "text": "회의를 시작하겠습니다"},
                {"start": 2.8, "end": 3.5, "text": "첫 번째 안건입니다"},
            ],
        ]

        merged = transcribe.merge_chunk_segments(chunks)

        self.assertEqual(
            [segment["text"] for segment in merged],
            ["안녕하세요", "회의를 시작하겠습니다", "첫 번째 안건입니다"],
        )

    def test_format_transcript_collapses_long_repeated_short_words(self):
        merged = [
            {
                "speaker": "Speaker A",
                "start": 0.0,
                "end": 1.0,
                "text": "정말 정말 정말 정말 좋은 것 같아요",
            }
        ]

        result = transcribe.format_transcript(merged)

        self.assertIn("정말 정말 좋은 것 같아요", result)
        self.assertNotIn("정말 정말 정말 정말", result)

    def test_format_transcript_collapses_repeated_single_syllables(self):
        merged = [
            {
                "speaker": "Speaker A",
                "start": 0.0,
                "end": 1.0,
                "text": "그그그 이건 다시 보죠",
            }
        ]

        result = transcribe.format_transcript(merged)

        self.assertIn("그 이건 다시 보죠", result)
        self.assertNotIn("그그그", result)


if __name__ == "__main__":
    unittest.main()
