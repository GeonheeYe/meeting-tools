# ~/meeting_tools/transcribe.py
"""
Whisper로 오디오 파일을 텍스트로 변환한다.
pyannote로 화자를 분리하고 두 결과를 병합한다.
"""
import os
import platform
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
_WHISPER_MODEL = None
_PYANNOTE_PIPELINE = None


def get_pyannote_pipeline(hf_token: str):
    """pyannote Pipeline을 1회만 로드하고 재사용한다."""
    global _PYANNOTE_PIPELINE
    if _PYANNOTE_PIPELINE is None:
        _PYANNOTE_PIPELINE = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token,
        )
    return _PYANNOTE_PIPELINE


def _is_apple_silicon() -> bool:
    """Apple Silicon(M시리즈) 여부 감지."""
    return platform.machine() == "arm64" or platform.processor() == "arm"


def get_whisper_model() -> WhisperModel:
    """Whisper 모델을 1회만 로드하고 재사용한다."""
    global _WHISPER_MODEL
    if _WHISPER_MODEL is None:
        _WHISPER_MODEL = WhisperModel("large-v3", device="cpu", compute_type="int8")
    return _WHISPER_MODEL


def build_initial_prompt(
    meeting_title: Optional[str] = None,
    context: Optional[str] = None,
    term_metadata: Optional[dict] = None,
) -> Optional[str]:
    """회의 제목과 문서 용어를 구조화해 Whisper 프롬프트를 만든다."""
    parts = []
    if meeting_title:
        parts.append(f"회의 주제: {meeting_title}")
    if context:
        parts.append(f"기본 컨텍스트: {context}")

    term_metadata = term_metadata or {}
    priority_terms = term_metadata.get("priority_terms", [])
    if priority_terms:
        parts.append(f"핵심 용어: {', '.join(priority_terms[:10])}")

    alias_map = term_metadata.get("alias_map", {})
    if alias_map:
        alias_pairs = [f"{alias} -> {canonical}" for alias, canonical in list(alias_map.items())[:10]]
        parts.append(f"표기 참고: {'; '.join(alias_pairs)}")

    if not parts:
        return None
    return "\n".join(parts)


def _run_whisper_faster(audio_path: str, initial_prompt: Optional[str] = None) -> list[dict]:
    """faster-whisper로 STT 실행 (non-Apple Silicon)."""
    print("Whisper STT 실행 중... (처음 실행 시 모델 다운로드 필요)")
    model = get_whisper_model()
    segments_gen, _ = model.transcribe(
        audio_path,
        language="ko",
        condition_on_previous_text=False,
        compression_ratio_threshold=2.4,
        no_speech_threshold=0.5,
        log_prob_threshold=-1.0,
        initial_prompt=initial_prompt,
        vad_filter=False,
    )
    segments = [
        {"start": seg.start, "end": seg.end, "text": seg.text.strip()}
        for seg in segments_gen
        if seg.text.strip()
    ]
    print(f"STT 완료: {len(segments)}개 세그먼트")
    return segments


def _run_whisper_mlx(audio_path: str, initial_prompt: Optional[str] = None) -> list[dict]:
    """mlx-whisper로 STT 실행 (Apple Silicon — Metal GPU 가속)."""
    import mlx_whisper
    print("Whisper STT 실행 중 (mlx-whisper, Metal 가속)...")
    result = mlx_whisper.transcribe(
        audio_path,
        path_or_hf_repo="mlx-community/whisper-large-v3-mlx",
        language="ko",
        initial_prompt=initial_prompt,
        condition_on_previous_text=False,
        no_speech_threshold=0.5,
        compression_ratio_threshold=2.4,
    )
    segments = [
        {"start": seg["start"], "end": seg["end"], "text": seg["text"].strip()}
        for seg in result.get("segments", [])
        if seg["text"].strip()
    ]
    print(f"STT 완료: {len(segments)}개 세그먼트")
    return segments


def run_whisper(audio_path: str, initial_prompt: Optional[str] = None) -> list[dict]:
    """STT 실행. Apple Silicon이면 mlx-whisper, 아니면 faster-whisper 사용."""
    if _is_apple_silicon():
        return _run_whisper_mlx(audio_path, initial_prompt)
    return _run_whisper_faster(audio_path, initial_prompt)


def offset_segments(segments: list[dict], chunk_start: float) -> list[dict]:
    """청크 내부 타임스탬프를 전체 오디오 기준으로 변환한다."""
    shifted = []
    for seg in segments:
        shifted.append({
            "start": seg["start"] + chunk_start,
            "end": seg["end"] + chunk_start,
            "text": seg["text"],
        })
    return shifted


def run_diarization(audio_path: str, num_speakers: Optional[int] = None) -> list[tuple]:
    """pyannote로 화자 분리. [(speaker, start, end), ...] 반환."""
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN 환경변수가 설정되지 않았습니다.")

    print("화자 분리 실행 중...")
    pipeline = get_pyannote_pipeline(hf_token)  # 캐싱된 pipeline 사용
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


def deduplicate_overlap_text(first_lines: list[str], second_lines: list[str]) -> list[str]:
    """오버랩 구간의 동일 문장을 단순 중복 제거한다."""
    merged = list(first_lines)
    seen = {line.strip() for line in first_lines if line.strip()}
    for line in second_lines:
        normalized = line.strip()
        if normalized and normalized not in seen:
            merged.append(line)
            seen.add(normalized)
    return merged


def merge_chunk_segments(chunks: list[list[dict]]) -> list[dict]:
    """청크별 세그먼트를 시간순으로 합치고 단순 중복을 제거한다."""
    merged = sorted(
        [segment for chunk in chunks for segment in chunk],
        key=lambda item: (item["start"], item["end"]),
    )

    deduplicated = []
    for seg in merged:
        if deduplicated and deduplicated[-1]["text"].strip() == seg["text"].strip():
            continue
        deduplicated.append(seg)
    return deduplicated


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
