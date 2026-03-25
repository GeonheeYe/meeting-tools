# ~/meeting_tools/transcribe.py
"""
Whisper로 오디오 파일을 텍스트로 변환한다.
pyannote로 화자를 분리하고 두 결과를 병합한다.
"""
import os
import platform
import re
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
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
DEFAULT_MAX_CHUNK_SEC = 90.0
DEFAULT_OVERLAP_SEC = 3.0
WHISPER_CONDITION_ON_PREVIOUS_TEXT = False
WHISPER_NO_SPEECH_THRESHOLD = 0.4


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
        condition_on_previous_text=WHISPER_CONDITION_ON_PREVIOUS_TEXT,
        compression_ratio_threshold=2.4,
        no_speech_threshold=WHISPER_NO_SPEECH_THRESHOLD,
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
        condition_on_previous_text=WHISPER_CONDITION_ON_PREVIOUS_TEXT,
        no_speech_threshold=WHISPER_NO_SPEECH_THRESHOLD,
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


def _get_audio_duration(audio_path: str) -> Optional[float]:
    """ffprobe로 오디오 길이를 구한다. 실패하면 None."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                audio_path,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None

    try:
        return float(result.stdout.strip())
    except ValueError:
        return None


def should_use_chunking(audio_path: str, max_chunk_sec: float = DEFAULT_MAX_CHUNK_SEC) -> bool:
    """해당 오디오가 chunked STT 대상인지 판단한다."""
    duration_sec = _get_audio_duration(audio_path)
    return duration_sec is not None and duration_sec > max_chunk_sec


def _build_chunk_ranges(duration_sec: float, max_chunk_sec: float, overlap_sec: float) -> list[tuple[float, float]]:
    """길이를 기준으로 청크 구간 목록을 만든다."""
    if duration_sec <= max_chunk_sec:
        return [(0.0, duration_sec)]

    ranges = []
    start = 0.0
    while start < duration_sec:
        end = min(start + max_chunk_sec, duration_sec)
        ranges.append((start, end))
        if end >= duration_sec:
            break
        start = max(end - overlap_sec, start + 0.1)
    return ranges


MIN_CHUNK_SEC = 15.0


def _build_vad_chunk_ranges(
    speech_timestamps: list[dict],
    duration_sec: float,
    max_chunk_sec: float,
) -> list[tuple[float, float]]:
    """VAD 타임스탬프 기반으로 무음 구간에서 청크를 나눈다."""
    if duration_sec <= max_chunk_sec:
        return [(0.0, duration_sec)]

    # 무음 구간 중간점 추출 (최소 0.3초 이상 무음)
    silence_midpoints = []
    for i in range(len(speech_timestamps) - 1):
        gap_start = speech_timestamps[i]["end"] / 16000
        gap_end = speech_timestamps[i + 1]["start"] / 16000
        if gap_end - gap_start >= 0.3:
            silence_midpoints.append((gap_start + gap_end) / 2)

    if not silence_midpoints:
        return _build_chunk_ranges(duration_sec, max_chunk_sec, DEFAULT_OVERLAP_SEC)

    # 탐욕적 분할: max_chunk_sec 이내에서 가장 늦은 무음 지점에서 자르기
    ranges = []
    chunk_start = 0.0

    while chunk_start < duration_sec - 1.0:
        chunk_end_target = chunk_start + max_chunk_sec
        if chunk_end_target >= duration_sec:
            ranges.append((chunk_start, duration_sec))
            break

        best_split = None
        for mp in silence_midpoints:
            if chunk_start < mp <= chunk_end_target:
                best_split = mp

        if best_split:
            ranges.append((chunk_start, best_split))
            chunk_start = best_split
        else:
            # 무음 지점 없으면 고정 분할 (fallback)
            ranges.append((chunk_start, chunk_end_target))
            chunk_start = chunk_end_target

    # 짧은 청크를 인접 청크와 합치기
    merged = [ranges[0]]
    for r in ranges[1:]:
        prev_len = merged[-1][1] - merged[-1][0]
        curr_len = r[1] - r[0]
        if prev_len < MIN_CHUNK_SEC or curr_len < MIN_CHUNK_SEC:
            merged[-1] = (merged[-1][0], r[1])
        else:
            merged.append(r)

    return merged


def _extract_audio_chunk(audio_path: str, start_sec: float, end_sec: float, chunk_index: int) -> str:
    """ffmpeg로 청크를 임시 wav로 추출한다."""
    duration = max(end_sec - start_sec, 0.1)
    tmp_dir = Path(tempfile.gettempdir())
    fd, out_str = tempfile.mkstemp(
        prefix=f"meeting_chunk_{Path(audio_path).stem}_{chunk_index}_",
        suffix=".wav",
        dir=tmp_dir,
    )
    os.close(fd)
    Path(out_str).unlink(missing_ok=True)
    out_path = Path(out_str)
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-ss",
            f"{start_sec:.3f}",
            "-t",
            f"{duration:.3f}",
            "-i",
            audio_path,
            "-ar",
            "16000",
            "-ac",
            "1",
            str(out_path),
            "-y",
        ],
        check=True,
        capture_output=True,
    )
    return str(out_path)


def run_whisper_chunked(
    audio_path: str,
    initial_prompt: Optional[str] = None,
    max_chunk_sec: float = DEFAULT_MAX_CHUNK_SEC,
    overlap_sec: float = DEFAULT_OVERLAP_SEC,
    speech_timestamps: Optional[list[dict]] = None,
) -> list[dict]:
    """긴 오디오는 청크로 나눠 Whisper를 실행하고 병합한다."""
    duration_sec = _get_audio_duration(audio_path)
    if duration_sec is None or duration_sec <= max_chunk_sec:
        return run_whisper(audio_path, initial_prompt=initial_prompt)

    # VAD 기반 청크 분할 (speech_timestamps가 있으면 사용)
    if speech_timestamps:
        chunk_ranges = _build_vad_chunk_ranges(speech_timestamps, duration_sec, max_chunk_sec)
        print(f"VAD 기반 청크 분할: {len(chunk_ranges)}개")
    else:
        chunk_ranges = _build_chunk_ranges(duration_sec, max_chunk_sec, overlap_sec)
        print(f"고정 시간 청크 분할: {len(chunk_ranges)}개")

    all_segments = []
    prev_chunk_text = ""

    for index, (start_sec, end_sec) in enumerate(chunk_ranges):
        # 이전 청크 텍스트를 initial_prompt에 추가 (문맥 연결)
        chunk_prompt = initial_prompt or ""
        if prev_chunk_text:
            tail = prev_chunk_text[-200:]
            chunk_prompt = f"{chunk_prompt}\n{tail}".strip() if chunk_prompt else tail

        chunk_path = None
        try:
            chunk_path = _extract_audio_chunk(audio_path, start_sec, end_sec, index)
            chunk_segments = run_whisper(chunk_path, initial_prompt=chunk_prompt or None)

            # 이 청크 텍스트를 다음 청크 프롬프트용으로 저장
            prev_chunk_text = " ".join(s["text"] for s in chunk_segments)

            all_segments.append(offset_segments(chunk_segments, start_sec))
        finally:
            if chunk_path:
                Path(chunk_path).unlink(missing_ok=True)

    return merge_chunk_segments(all_segments)


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


def _text_similarity(a: str, b: str) -> float:
    """두 텍스트의 유사도를 0~1로 반환한다 (편집거리 기반)."""
    a, b = a.strip(), b.strip()
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    max_len = max(len(a), len(b))
    # 길이 차이가 크면 빠르게 비유사 판정
    if abs(len(a) - len(b)) / max_len > 0.5:
        return 0.0
    # 간단한 편집거리 (DP)
    m, n = len(a), len(b)
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        curr = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = curr
    return 1.0 - prev[n] / max_len


def merge_chunk_segments(chunks: list[list[dict]]) -> list[dict]:
    """청크별 세그먼트를 시간순으로 합치고 유사 중복을 제거한다."""
    merged = sorted(
        [segment for chunk in chunks for segment in chunk],
        key=lambda item: (item["start"], item["end"]),
    )

    deduplicated = []
    for seg in merged:
        if deduplicated:
            prev = deduplicated[-1]
            # 시간이 겹치거나 인접 + 텍스트 유사도 80% 이상이면 중복
            time_overlap = prev["end"] > seg["start"] - 1.0
            if time_overlap and _text_similarity(prev["text"], seg["text"]) >= 0.8:
                continue
        deduplicated.append(seg)
    return deduplicated


def transcribe(
    audio_path: str,
    num_speakers: Optional[int] = None,
    initial_prompt: Optional[str] = None,
    skip_diarization: bool = False,
    speech_timestamps: Optional[list[dict]] = None,
) -> list[dict]:
    """전체 파이프라인: STT + (선택적) 화자분리 병렬 실행 + 병합."""
    if skip_diarization:
        segments = run_whisper_chunked(audio_path, initial_prompt=initial_prompt, speech_timestamps=speech_timestamps)
        return [{"speaker": "Speaker A", "start": s["start"],
                 "end": s["end"], "text": s["text"]} for s in segments]

    # STT와 화자분리를 병렬로 실행 (두 작업은 독립적)
    with ThreadPoolExecutor(max_workers=2) as executor:
        stt_future = executor.submit(
            run_whisper_chunked, audio_path, initial_prompt,
            DEFAULT_MAX_CHUNK_SEC, DEFAULT_OVERLAP_SEC, speech_timestamps,
        )
        diar_future = executor.submit(run_diarization, audio_path, num_speakers)

    segments = stt_future.result()
    turns = diar_future.result()
    return merge(segments, turns)


def _collapse_runaway_syllables(text: str) -> str:
    """붙은 짧은 음절 반복 노이즈를 1회로 줄인다."""
    return re.sub(r"\b([가-힣A-Za-z])\1{2,}\b", r"\1", text)


def _collapse_runaway_short_tokens(text: str) -> str:
    """짧은 단어가 과도하게 반복되면 2회까지만 남긴다."""
    return re.sub(r"\b([가-힣A-Za-z]{1,3})(?:\s+\1){2,}\b", r"\1 \1", text)


def _remove_hallucination_fragments(text: str) -> str:
    """Whisper 환각으로 생성된 비한국어 깨진 토큰을 제거한다."""
    # 라틴 특수문자 조합 (예: "장łe의", "adjective은")
    cleaned = re.sub(r"[가-힣A-Za-z]*[ąćęłńóśźżĄĆĘŁŃÓŚŹŻ]+[가-힣A-Za-z]*", "", text)
    # 한국어 문맥에서 영단어만 반복되는 환각 (예: "them", "the")
    cleaned = re.sub(r"\b(them|the|and|of|is|to|in|it|for|that|this|with|you|are|was|have|has)\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleaned


def _is_hallucination_segment(text: str) -> bool:
    """세그먼트 전체가 환각인지 판단한다."""
    cleaned = text.strip()
    if not cleaned:
        return True
    # 한국어가 전혀 없고 의미 있는 내용도 없는 경우
    has_korean = bool(re.search(r"[가-힣]", cleaned))
    if not has_korean and len(cleaned) < 20:
        return True
    return False


def clean_repetition_noise(text: str) -> str:
    """STT가 만든 비정상 반복 노이즈를 보수적으로 정리한다."""
    cleaned = _collapse_runaway_syllables(text)
    cleaned = _collapse_runaway_short_tokens(cleaned)
    cleaned = _remove_hallucination_fragments(cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleaned


def format_transcript(merged: list[dict]) -> str:
    """대화록을 읽기 좋은 텍스트로 변환."""
    # 짧은 단독 발화("네", "음", "그" 등) 연속 반복 제거
    _FILLER = {"네", "네.", "음", "음.", "그", "그.", "응", "응.", "어", "어.", "아", "아."}

    lines = []
    prev_speaker = None
    prev_text = None
    repeat_count = 0

    for item in merged:
        text = clean_repetition_noise(item["text"])
        speaker = item["speaker"]

        if not text:
            continue

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
