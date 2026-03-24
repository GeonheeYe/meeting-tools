# Meeting 스킬 개선 설계

**날짜**: 2026-03-19
**범위**: SKILL.md UX 개선 + meeting_tools 성능 개선

---

## 개요

두 가지 축으로 개선한다:
1. **UX**: 진행 상태 피드백 + Notion 업로드 전 검토 단계
2. **성능**: mlx-whisper, 병렬 실행, pyannote 캐싱, VAD 이중 read 제거

---

## 변경 1: SKILL.md — 진행 상태 피드백

Step 2(파이프라인 실행) 앞뒤에 메시지 출력 지시 추가.

- 실행 전: `"⏳ STT 파이프라인을 시작합니다. 오디오 길이에 따라 수 분 소요될 수 있습니다..."`
- 실행 후: `"✅ 파이프라인 완료"`

## 변경 2: SKILL.md — Step 4.5 신설 (업로드 전 검토)

Step 4(요약 생성)와 Step 5(Notion 업로드) 사이에 삽입.

1. 터미널에 회의 요약 / 액션 아이템 / 주요 결정사항 출력
2. `AskUserQuestion`으로 확인:
   - **업로드** → Step 5 진행
   - **취소** → 종료 + JSON 파일 삭제

## 변경 3: transcribe.py — mlx-whisper 도입

`platform.processor()`로 Apple Silicon 감지:
- M시리즈(`arm`) → `mlx-whisper` 사용 (Metal GPU 가속, 예상 3~5x 향상)
- 그 외 → 기존 `faster-whisper` 유지

```python
import platform
import subprocess

def _is_apple_silicon() -> bool:
    return platform.processor() == "arm" or platform.machine() == "arm64"
```

mlx-whisper API:
```python
import mlx_whisper
result = mlx_whisper.transcribe(audio_path, path_or_hf_repo="mlx-community/whisper-large-v3-mlx")
```

세그먼트 구조는 faster-whisper와 동일하게 맞춤.

## 변경 4: transcribe.py — Whisper + pyannote 병렬 실행

화자분리가 있을 때(`skip_diarization=False`)만 해당.

```python
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=2) as executor:
    stt_future = executor.submit(run_whisper, audio_path, initial_prompt)
    diar_future = executor.submit(run_diarization, audio_path, num_speakers)

segments = stt_future.result()
turns = diar_future.result()
```

총 처리 시간: `STT시간 + 화자분리시간` → `max(STT시간, 화자분리시간)`

## 변경 5: transcribe.py — pyannote 모델 캐싱

```python
_PYANNOTE_PIPELINE = None

def get_pyannote_pipeline(hf_token: str):
    global _PYANNOTE_PIPELINE
    if _PYANNOTE_PIPELINE is None:
        _PYANNOTE_PIPELINE = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token,
        )
    return _PYANNOTE_PIPELINE
```

`run_diarization()`에서 `Pipeline.from_pretrained()` 직접 호출 → `get_pyannote_pipeline()` 호출로 교체.

## 변경 6: pipeline.py — VAD 이중 read 제거

현재: `get_vad_speech_segments()`와 `run_vad()` 각각 `read_audio()` 호출.

개선: `run_vad()` 하나로 통합, wav를 한 번만 읽고 speech_timestamps와 audio_chunks를 모두 처리.

```python
def run_vad(path: Path) -> Path:
    wav = read_audio(str(path), sampling_rate=16000)  # 한 번만 읽기
    model = load_silero_vad()
    speech_timestamps = get_speech_timestamps(wav, model, ...)
    # ... 이후 처리
```

`get_vad_speech_segments()` 함수는 제거 (내부에서만 쓰임).

## 변경 7: requirements.txt

- `openai-whisper` 제거 (코드에서 미사용)
- `mlx-whisper` 추가

---

## 파일별 변경 요약

| 파일 | 변경 |
|------|------|
| `SKILL.md` | 진행 메시지 + Step 4.5 추가 |
| `transcribe.py` | mlx-whisper, 병렬 실행, pyannote 캐싱 |
| `pipeline.py` | VAD 이중 read 제거 |
| `requirements.txt` | openai-whisper 제거, mlx-whisper 추가 |
