# ~/meeting_tools/transcribe.py
"""
Whisper로 오디오 파일을 텍스트로 변환한다.
pyannote로 화자를 분리하고 두 결과를 병합한다.
"""
import os
from pathlib import Path
from typing import Optional

import torch
from dotenv import load_dotenv

# PyTorch 2.6에서 weights_only 기본값이 True로 변경되어 pyannote 모델 로딩 실패
# pyannote는 신뢰된 소스이므로 weights_only=False로 강제 (2.5 이전 동작)
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False  # setdefault 대신 강제 덮어쓰기
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from faster_whisper import WhisperModel
from pyannote.audio import Pipeline

load_dotenv(Path(__file__).parent / ".env")


def run_whisper(audio_path: str, initial_prompt: Optional[str] = None) -> list[dict]:
    """Whisper STT 실행. 타임스탬프 포함 세그먼트 반환."""
    print("Whisper STT 실행 중... (처음 실행 시 모델 다운로드 필요)")
    # faster-whisper: openai-whisper 대비 4배 빠름, large-v3로 정확도 향상
    # CPU + int8: Mac에서 CTranslate2 최적 조합
    model = WhisperModel("large-v3", device="cpu", compute_type="int8")
    segments_gen, _ = model.transcribe(
        audio_path,
        language="ko",
        condition_on_previous_text=False,   # 연쇄 hallucination 방지
        compression_ratio_threshold=2.4,    # 반복 텍스트 세그먼트 제거
        no_speech_threshold=0.5,            # 무음 구간 텍스트 생성 방지
        log_prob_threshold=-1.0,            # 확신 낮은 세그먼트 제거
        initial_prompt=initial_prompt,      # 기술 용어 힌트로 hallucination 방지
        vad_filter=False,                   # 별도 VAD 처리 완료 → 비활성화
    )
    segments = [
        {"start": seg.start, "end": seg.end, "text": seg.text.strip()}
        for seg in segments_gen
        if seg.text.strip()
    ]
    print(f"STT 완료: {len(segments)}개 세그먼트")
    return segments


def run_diarization(audio_path: str, num_speakers: Optional[int] = None) -> list[tuple]:
    """pyannote로 화자 분리. [(speaker, start, end), ...] 반환."""
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN 환경변수가 설정되지 않았습니다.")

    print("화자 분리 실행 중...")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token,
    )
    # 화자 수를 알고 있으면 힌트로 전달 → 분리 품질 향상
    diarize_kwargs = {}
    if num_speakers:
        diarize_kwargs["num_speakers"] = num_speakers
        print(f"화자 수 힌트: {num_speakers}명")
    diarization = pipeline(audio_path, **diarize_kwargs)

    turns = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        turns.append((speaker, turn.start, turn.end))
    print(f"화자 분리 완료: {len(set(t[0] for t in turns))}명 감지")
    return turns


def _find_speaker(start: float, end: float, turns: list[tuple]) -> str:
    """세그먼트 시간과 가장 많이 겹치는 화자 반환."""
    overlap: dict[str, float] = {}
    for speaker, t_start, t_end in turns:
        o = max(0.0, min(end, t_end) - max(start, t_start))
        if o > 0:
            overlap[speaker] = overlap.get(speaker, 0) + o
    if not overlap:
        return "Unknown"
    return max(overlap, key=overlap.get)


def merge(segments: list[dict], turns: list[tuple]) -> list[dict]:
    """STT 세그먼트에 화자 정보를 붙인다."""
    speaker_map: dict[str, str] = {}

    def _label(n: int) -> str:
        # A-Z, 이후 AA, AB...
        if n < 26:
            return f"Speaker {chr(65 + n)}"
        return f"Speaker {chr(65 + n // 26 - 1)}{chr(65 + n % 26)}"

    result = []
    for seg in segments:
        raw = _find_speaker(seg["start"], seg["end"], turns)
        if raw not in speaker_map:
            speaker_map[raw] = _label(len(speaker_map))
        result.append({
            "speaker": speaker_map[raw],
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"],
        })
    return result


def transcribe(
    audio_path: str,
    num_speakers: Optional[int] = None,
    initial_prompt: Optional[str] = None,
    skip_diarization: bool = False,
) -> list[dict]:
    """전체 파이프라인: STT + (선택적) 화자분리 + 병합."""
    segments = run_whisper(audio_path, initial_prompt=initial_prompt)

    if skip_diarization:
        # 화자 분리 없이 Speaker A 단일 레이블로 반환
        return [{"speaker": "Speaker A", "start": s["start"],
                 "end": s["end"], "text": s["text"]} for s in segments]

    turns = run_diarization(audio_path, num_speakers=num_speakers)
    return merge(segments, turns)


def format_transcript(merged: list[dict]) -> str:
    """대화록을 읽기 좋은 텍스트로 변환."""
    # 짧은 단독 발화("네", "음", "그" 등) 연속 반복 제거
    _FILLER = {"네", "네.", "음", "음.", "그", "그.", "응", "응.", "어", "어.", "아", "아."}

    lines = []
    prev_speaker = None
    prev_text = None
    repeat_count = 0

    for item in merged:
        text = item["text"]
        speaker = item["speaker"]

        # 동일 텍스트 3회 이상 연속 반복 시 스킵
        if text == prev_text:
            repeat_count += 1
            if repeat_count >= 2:
                continue
        else:
            repeat_count = 0

        # filler 단어만 있는 세그먼트는 화자당 1회만 허용
        if text in _FILLER and speaker == prev_speaker:
            prev_text = text
            continue

        if speaker != prev_speaker:
            lines.append(f"\n[{speaker}]")
            prev_speaker = speaker

        lines.append(text)
        prev_text = text

    return "\n".join(lines).strip()


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("사용법: python transcribe.py <오디오파일>")
        sys.exit(1)
    merged = transcribe(sys.argv[1])
    print("\n=== 대화록 ===")
    print(format_transcript(merged))
