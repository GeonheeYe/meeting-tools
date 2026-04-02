# ~/meeting_tools/pipeline.py
"""
전체 파이프라인 실행:
오디오 파일 → 오디오 보정 → STT → 화자분리 → 병합 → JSON 저장
요약 및 Notion 업로드는 /meeting 스킬이 직접 처리한다.
"""
import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

from context_loader import load as load_context
from transcribe import format_transcript, should_use_chunking, transcribe

# soundfile이 직접 읽지 못하는 포맷 → ffmpeg로 wav 변환
_NEEDS_CONVERSION = {".m4a", ".mp3", ".mp4", ".aac", ".ogg", ".flac"}
ENHANCED_AUDIO_FILTER = "loudnorm=I=-16:LRA=7:TP=-1,dynaudnorm=f=120:g=31,acompressor=threshold=0.05:ratio=4:attack=10:release=150:makeup=4"


def _to_wav(path: Path) -> Path:
    """ffmpeg로 16kHz mono wav로 변환. 변환된 임시 파일 경로 반환."""
    fd, out_str = tempfile.mkstemp(prefix=f"{path.stem}_converted_", suffix=".wav", dir="/tmp")
    os.close(fd)
    Path(out_str).unlink(missing_ok=True)
    out = Path(out_str)
    subprocess.run(
        ["ffmpeg", "-i", str(path), "-ar", "16000", "-ac", "1", str(out), "-y"],
        check=True, capture_output=True,
    )
    return out


def enhance_audio_for_stt(path: Path) -> Path:
    """STT용 오디오 보정본을 원본 옆에 저장한다."""
    out = path.with_name(f"{path.stem}_enhanced.wav")
    subprocess.run(
        [
            "ffmpeg",
            "-i",
            str(path),
            "-af",
            ENHANCED_AUDIO_FILTER,
            "-ar",
            "16000",
            "-ac",
            "1",
            str(out),
            "-y",
        ],
        check=True,
        capture_output=True,
    )
    return out


def normalize_terms(transcript: str, term_metadata: Optional[dict]) -> str:
    """문서 근거가 있는 별칭만 정규 표현식 없이 보수적으로 치환한다."""
    if not term_metadata:
        return transcript

    alias_map = term_metadata.get("alias_map", {})
    normalized = transcript
    for alias, canonical in sorted(alias_map.items(), key=lambda item: len(item[0]), reverse=True):
        if not alias or alias == canonical:
            continue
        pattern = re.compile(rf"(?<![A-Za-z0-9]){re.escape(alias)}(?![A-Za-z0-9])", flags=re.IGNORECASE)
        normalized = pattern.sub(canonical, normalized)
    return normalized


def run(
    audio_path: str,
    title: Optional[str] = None,
    num_speakers: Optional[int] = None,
    context: Optional[str] = None,
    doc_paths: Optional[list] = None,
) -> str:
    """파이프라인 실행. 대화록 JSON 파일 경로 반환."""
    original_path = Path(audio_path).expanduser().resolve()
    if not original_path.exists():
        raise FileNotFoundError(f"오디오 파일을 찾을 수 없습니다: {original_path}")

    # soundfile이 지원하지 않는 포맷은 wav로 변환
    converted = None
    path = original_path
    if path.suffix.lower() in _NEEDS_CONVERSION:
        print(f"{path.suffix} 포맷 감지 → wav로 변환 중...")
        converted = _to_wav(path)
        path = converted

    # 제목 자동 생성
    ts = datetime.now().strftime("%Y-%m-%d")
    if not title:
        title = f"[{ts}] 회의"

    # 참고 문서 로드 → initial_prompt 강화
    doc_content = ""
    term_metadata = {"canonical_terms": [], "priority_terms": [], "alias_map": {}}
    agenda_items = []
    if doc_paths:
        print(f"참고 문서 로드 중: {len(doc_paths)}개")
        try:
            key_terms, doc_content, term_metadata, agenda_items = load_context(doc_paths)
        except Exception as exc:
            print(f"참고 문서 로드 실패, 기본 STT로 계속 진행: {exc}")
            key_terms = ""
        if key_terms:
            context = f"{key_terms}, {context}" if context else key_terms

    print(f"\n{'='*50}")
    print(f"파이프라인 시작: {path.name}")
    if num_speakers:
        print(f"화자 수: {num_speakers}명")
    if context:
        print(f"컨텍스트: {context[:80]}...")
    print(f"{'='*50}\n")

    # STT + 화자 분리 (num_speakers 없으면 pyannote 생략)
    if num_speakers is None:
        print("화자 분리 생략 (--speakers 미지정)")

    stt_input_path = path
    audio_enhanced = False
    try:
        stt_input_path = enhance_audio_for_stt(path)
        audio_enhanced = stt_input_path != path
    except Exception as exc:
        print(f"오디오 보정 실패, 원본 오디오로 계속 진행: {exc}")
        stt_input_path = path

    chunking_applied = should_use_chunking(str(stt_input_path))

    if num_speakers is None:
        print("화자 분리 생략 (--speakers 미지정)")
    merged = transcribe(
        str(stt_input_path),
        num_speakers=num_speakers,
        initial_prompt=context,
        skip_diarization=num_speakers is None,
    )
    transcript = normalize_terms(format_transcript(merged), term_metadata)
    speaker_count = num_speakers or len(set(item["speaker"] for item in merged))

    # 결과 JSON 저장 (요약/Notion 업로드는 스킬이 처리)
    result = {
        "title": title,
        "date": ts,
        "speaker_count": speaker_count,
        "transcript": transcript,
        "doc_content": doc_content,  # Claude 교정용, 없으면 ""
        "agenda_items": agenda_items,  # 안건 항목 목록, 없으면 []
        "vad_applied": False,
        "chunking_applied": chunking_applied,
        "source_audio_path": str(original_path),
        "stt_audio_path": str(stt_input_path),
        "term_metadata_applied": bool(term_metadata.get("priority_terms") or term_metadata.get("alias_map")),
        "audio_enhanced": audio_enhanced,
    }
    # 결과 JSON은 원본 오디오 파일과 같은 디렉토리에 저장
    save_dir = original_path.parent
    save_path = save_dir / f"{original_path.stem}.json"
    save_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"STT 결과 저장: {save_path}")

    # 임시 파일 정리 (enhanced.wav는 유지)
    if converted and converted.exists():
        converted.unlink()

    print(f"\n{'='*50}")
    print(f"처리 완료. 결과: {save_path}")
    print(f"{'='*50}\n")
    return str(save_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="회의 오디오 처리 파이프라인")
    parser.add_argument("audio", help="오디오 파일 경로")
    parser.add_argument("title", nargs="?", default=None, help="회의 제목")
    parser.add_argument("--speakers", type=int, default=None, help="화자 수 (pyannote 힌트)")
    parser.add_argument("--context", default=None, help="회의 컨텍스트/키워드 (Whisper initial_prompt)")
    parser.add_argument("--docs", nargs="*", default=None, help="참고 문서 경로 목록 (PDF, TXT, DOCX)")
    args = parser.parse_args()
    print(run(args.audio, args.title, args.speakers, args.context, args.docs))
