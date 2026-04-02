"""
파라미터 그리드 서치 eval 스크립트.
테스트 오디오에 대해 여러 파라미터 조합으로 STT를 실행하고
Claude-as-judge로 채점해 최적값을 찾는다.

사용법:
  python3 eval_params.py <오디오파일경로>
"""
import importlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional


# --- 테스트할 파라미터 그리드 ---
GRID = [
    {"no_speech_threshold": 0.3, "max_chunk_sec": 90,  "overlap_sec": 3},
    {"no_speech_threshold": 0.4, "max_chunk_sec": 90,  "overlap_sec": 3},  # 현재 기본값
    {"no_speech_threshold": 0.5, "max_chunk_sec": 90,  "overlap_sec": 3},
    {"no_speech_threshold": 0.6, "max_chunk_sec": 90,  "overlap_sec": 3},
    {"no_speech_threshold": 0.5, "max_chunk_sec": 60,  "overlap_sec": 3},
    {"no_speech_threshold": 0.5, "max_chunk_sec": 120, "overlap_sec": 3},
    {"no_speech_threshold": 0.5, "max_chunk_sec": 90,  "overlap_sec": 5},
]

# Whisper hallucination 패턴 (자주 등장하는 가짜 자막)
HALLUCINATION_PATTERNS = [
    "자막 제공", "영상 제공", "광고를 포함", "구독", "좋아요", "알림 설정",
    "시청해주셔서 감사합니다", "다음 영상에서", "MBC", "KBS", "SBS",
    "자막은", "번역", "이 영상은", "영상을 시청",
]


def count_hallucinations(text: str) -> int:
    """hallucination 패턴 등장 횟수를 센다."""
    count = 0
    for pattern in HALLUCINATION_PATTERNS:
        count += text.count(pattern)
    return count


def count_repetitions(text: str, min_len: int = 10) -> int:
    """동일한 구절(10자 이상)이 3회 이상 반복되는 케이스 수를 센다."""
    lines = [l.strip() for l in text.splitlines() if len(l.strip()) >= min_len]
    from collections import Counter
    counts = Counter(lines)
    return sum(1 for c in counts.values() if c >= 3)


def run_stt_with_params(audio_path: str, params: dict) -> Optional[list[dict]]:
    """주어진 파라미터로 mlx-whisper STT를 실행한다. 청킹 포함."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    import transcribe as tr
    import importlib

    # 상수를 monkey-patch로 교체
    tr.WHISPER_NO_SPEECH_THRESHOLD = params["no_speech_threshold"]
    tr.DEFAULT_MAX_CHUNK_SEC = float(params["max_chunk_sec"])
    tr.DEFAULT_OVERLAP_SEC = float(params["overlap_sec"])

    try:
        segments = tr.run_whisper(audio_path, initial_prompt=None)
        return segments
    except Exception as e:
        print(f"  STT 실패: {e}")
        return None


def format_transcript(segments: list[dict]) -> str:
    """세그먼트 리스트를 텍스트로 변환."""
    return "\n".join(seg["text"] for seg in segments)


def judge_heuristic(transcript: str) -> dict:
    """heuristic으로 transcript 품질을 채점한다. 0-10점."""
    halluc_count = count_hallucinations(transcript)
    repeat_count = count_repetitions(transcript)
    char_count = len(transcript)

    score = 10
    # hallucination 패턴 1건당 -2점
    score -= min(halluc_count * 2, 6)
    # 반복 구절 1건당 -1점
    score -= min(repeat_count, 3)
    # 내용이 너무 적으면 감점 (100자 미만)
    if char_count < 100:
        score -= 3
    elif char_count < 300:
        score -= 1

    reasons = []
    if halluc_count > 0:
        reasons.append(f"hallucination {halluc_count}건")
    if repeat_count > 0:
        reasons.append(f"반복 {repeat_count}건")
    if char_count < 100:
        reasons.append("내용 부족")

    reason = ", ".join(reasons) if reasons else "문제 없음"
    return {"score": max(0, score), "reason": reason}


def main():
    if len(sys.argv) < 2:
        print("사용법: python3 eval_params.py <오디오파일경로>")
        sys.exit(1)

    audio_path = sys.argv[1]
    if not Path(audio_path).exists():
        print(f"파일 없음: {audio_path}")
        sys.exit(1)

    print(f"=== 파라미터 그리드 서치 시작 ===")
    print(f"테스트 파일: {audio_path}")
    print(f"파라미터 조합: {len(GRID)}개\n")

    results = []

    for i, params in enumerate(GRID):
        label = (f"no_speech={params['no_speech_threshold']}, "
                 f"chunk={params['max_chunk_sec']}s, "
                 f"overlap={params['overlap_sec']}s")
        print(f"[{i+1}/{len(GRID)}] {label}")

        t0 = time.time()
        segments = run_stt_with_params(audio_path, params)
        elapsed = time.time() - t0

        if segments is None:
            print("  → 실패, 스킵\n")
            continue

        transcript = format_transcript(segments)
        halluc = count_hallucinations(transcript)
        reps = count_repetitions(transcript)
        print(f"  STT: {len(segments)}세그먼트, {len(transcript)}자, {elapsed:.1f}s")
        print(f"  hallucination={halluc}, 반복={reps}")

        judgment = judge_heuristic(transcript)
        score = judgment.get("score", 0)
        reason = judgment.get("reason", "")
        print(f"  점수: {score}/10 — {reason}\n")

        results.append({
            "params": params,
            "label": label,
            "segments": len(segments),
            "chars": len(transcript),
            "hallucinations": halluc,
            "repetitions": reps,
            "score": score,
            "reason": reason,
            "elapsed": round(elapsed, 1),
            "transcript": transcript,  # 전체 저장
        })

    if not results:
        print("유효한 결과 없음")
        return

    # 결과 정렬
    results.sort(key=lambda r: r["score"], reverse=True)

    print("\n" + "="*60)
    print("=== 결과 요약 (점수 내림차순) ===")
    print("="*60)
    for r in results:
        marker = "★ 최적" if r == results[0] else "  "
        print(f"{marker} {r['label']}")
        print(f"     점수: {r['score']}/10 | chars: {r['chars']} | "
              f"halluc: {r['hallucinations']} | 반복: {r['repetitions']}")
        print(f"     근거: {r['reason']}")
        print()

    best = results[0]
    print("="*60)
    print(f"최적 파라미터:")
    print(f"  WHISPER_NO_SPEECH_THRESHOLD = {best['params']['no_speech_threshold']}")
    print(f"  DEFAULT_MAX_CHUNK_SEC       = {best['params']['max_chunk_sec']}")
    print(f"  DEFAULT_OVERLAP_SEC         = {best['params']['overlap_sec']}")
    print("="*60)

    # JSON 저장
    out_path = Path(audio_path).parent / "eval_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n전체 결과 저장: {out_path}")


if __name__ == "__main__":
    main()
