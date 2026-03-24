# Meeting 스킬 개선 Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** meeting 스킬의 UX(진행 상태 피드백, 업로드 전 검토)와 성능(mlx-whisper, 병렬 실행, 캐싱, VAD I/O 중복 제거)을 개선한다.

**Architecture:** SKILL.md는 Claude가 따르는 프롬프트 파일이므로 텍스트 지시만 추가한다. meeting_tools 코드는 `transcribe.py`(STT·화자분리)와 `pipeline.py`(VAD·조율)를 각각 수정하고, 각 변경마다 unittest를 먼저 작성한 뒤 구현한다.

**Tech Stack:** Python 3.11+, faster-whisper, mlx-whisper (Apple Silicon), pyannote.audio 3.1, silero-vad, concurrent.futures

---

## 파일 구조

| 파일 | 역할 | 변경 |
|------|------|------|
| `~/.claude/skills/meeting/SKILL.md` | Claude 실행 프롬프트 | Step 2 메시지 + Step 4.5 추가 |
| `transcribe.py` | STT + 화자분리 | mlx-whisper, 병렬 실행, pyannote 캐싱 |
| `pipeline.py` | VAD + 조율 | VAD 이중 read 제거 |
| `requirements.txt` | 의존성 | openai-whisper 제거, mlx-whisper 추가 |
| `test_transcribe_model_cache.py` | 모델 캐싱 테스트 | pyannote 캐싱 테스트 추가 |
| `test_transcribe_mlx.py` | mlx-whisper 테스트 | 신규 생성 |
| `test_transcribe_parallel.py` | 병렬 실행 테스트 | 신규 생성 |
| `test_pipeline_vad.py` | VAD 테스트 | 신규 생성 |

---

## Chunk 1: SKILL.md UX 개선

### Task 1: Step 2 진행 상태 메시지 추가

**Files:**
- Modify: `~/.claude/skills/meeting/SKILL.md`

현재 Step 2는 pipeline.py 실행 명령만 있고 사용자에게 상태를 알려주지 않는다. 명령 실행 전후에 메시지 출력 지시를 추가한다.

- [ ] **Step 1: SKILL.md Step 2 섹션에 메시지 지시 추가**

`### Step 2: 파이프라인 실행 (VAD + STT + 선택적 화자분리)` 아래 명령 실행 전에 다음 줄 삽입:

```
다음 명령을 실행하기 전에 사용자에게 알린다:
"⏳ STT 파이프라인을 시작합니다. 오디오 길이에 따라 수 분 소요될 수 있습니다..."
```

명령 실행 후 JSON 파일 경로가 출력되면 다음 줄 추가:

```
완료 후 사용자에게 알린다:
"✅ 파이프라인 완료"
```

- [ ] **Step 2: 커밋**

```bash
cd ~/.claude/skills/meeting
git add SKILL.md
git commit -m "feat(skill): add pipeline progress messages to meeting skill"
```

---

### Task 2: Step 4.5 신설 — 업로드 전 검토

**Files:**
- Modify: `~/.claude/skills/meeting/SKILL.md`

Step 4(요약 생성)와 Step 5(Notion 업로드) 사이에 새 섹션을 추가한다. 사용자가 요약 내용을 확인하고 업로드 여부를 결정하도록 한다.

- [ ] **Step 1: Step 4와 Step 5 사이에 Step 4.5 삽입**

`### Step 4: Claude 교정 + 요약` 섹션 뒤, `### Step 5: Notion MCP로 페이지 생성` 섹션 앞에 다음 내용 삽입:

```markdown
### Step 4.5: 사용자 확인

요약 생성 후 Notion에 업로드하기 전에 내용을 터미널에 출력하고 사용자 확인을 받는다.

터미널에 다음 형식으로 출력한다:

~~~
---
📋 회의 요약
(요약 내용 출력)

✅ 액션 아이템
(액션 아이템 목록 출력)

🔑 주요 결정사항
(결정사항 목록 출력)
---
~~~

그 다음 AskUserQuestion 도구로 묻는다:
- 질문: "위 내용을 Notion에 업로드할까요?"
- 옵션 1: "업로드" → Step 5로 진행
- 옵션 2: "취소" → JSON 파일 삭제 후 종료 ("취소되었습니다." 안내)
```

- [ ] **Step 2: 커밋**

```bash
git add SKILL.md
git commit -m "feat(skill): add pre-upload review step to meeting skill"
```

---

## Chunk 2: transcribe.py 성능 개선

### Task 3: mlx-whisper 도입

**Files:**
- Modify: `transcribe.py`
- Create: `test_transcribe_mlx.py`

Apple Silicon에서 mlx-whisper를 사용해 Metal GPU 가속을 활성화한다. `run_whisper()`가 플랫폼을 감지해 자동으로 분기한다.

- [ ] **Step 1: 실패 테스트 작성**

`test_transcribe_mlx.py` 생성:

```python
import unittest
from unittest.mock import MagicMock, patch

import transcribe


class AppleSiliconDetectionTest(unittest.TestCase):
    def test_detects_apple_silicon_arm64(self):
        with patch("platform.machine", return_value="arm64"):
            self.assertTrue(transcribe._is_apple_silicon())

    def test_not_apple_silicon_x86(self):
        with patch("platform.machine", return_value="x86_64"):
            with patch("platform.processor", return_value="Intel"):
                self.assertFalse(transcribe._is_apple_silicon())


class RunWhisperDispatchTest(unittest.TestCase):
    def setUp(self):
        # 모델 캐시 초기화
        transcribe._WHISPER_MODEL = None

    def test_uses_mlx_on_apple_silicon(self):
        fake_segments = [{"start": 0.0, "end": 1.5, "text": "안녕하세요"}]
        with patch("transcribe._is_apple_silicon", return_value=True):
            with patch("transcribe._run_whisper_mlx", return_value=fake_segments) as mlx_fn:
                result = transcribe.run_whisper("audio.wav")
        mlx_fn.assert_called_once_with("audio.wav", None)
        self.assertEqual(result, fake_segments)

    def test_uses_faster_whisper_on_non_apple(self):
        fake_segments = [{"start": 0.0, "end": 1.5, "text": "안녕하세요"}]
        with patch("transcribe._is_apple_silicon", return_value=False):
            with patch("transcribe._run_whisper_faster", return_value=fake_segments) as fw_fn:
                result = transcribe.run_whisper("audio.wav", "컨텍스트")
        fw_fn.assert_called_once_with("audio.wav", "컨텍스트")
        self.assertEqual(result, fake_segments)

    def test_mlx_output_has_required_fields(self):
        """mlx-whisper 출력이 faster-whisper와 동일한 필드를 갖는지 확인."""
        mock_mlx_result = {
            "segments": [
                {"start": 0.0, "end": 2.0, "text": " 테스트입니다", "id": 0}
            ]
        }
        with patch("transcribe._is_apple_silicon", return_value=True):
            with patch("mlx_whisper.transcribe", return_value=mock_mlx_result):
                result = transcribe._run_whisper_mlx("audio.wav", None)
        self.assertEqual(len(result), 1)
        self.assertIn("start", result[0])
        self.assertIn("end", result[0])
        self.assertIn("text", result[0])
        self.assertEqual(result[0]["text"], "테스트입니다")  # strip() 확인


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
cd ~/meeting_tools && python -m pytest test_transcribe_mlx.py -v
```

Expected: `AttributeError: module 'transcribe' has no attribute '_is_apple_silicon'`

- [ ] **Step 3: transcribe.py에 구현 추가**

`transcribe.py` 상단 import에 추가:

```python
import platform
```

`_WHISPER_MODEL = None` 아래에 추가:

```python
def _is_apple_silicon() -> bool:
    """Apple Silicon(M시리즈) 여부 감지."""
    return platform.machine() == "arm64" or platform.processor() == "arm"
```

기존 `run_whisper()` 함수를 다음으로 교체:

```python
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
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
python -m pytest test_transcribe_mlx.py -v
```

Expected: 5개 PASS

- [ ] **Step 5: 커밋**

```bash
git add transcribe.py test_transcribe_mlx.py
git commit -m "perf(transcribe): add mlx-whisper support for Apple Silicon"
```

---

### Task 4: pyannote 모델 캐싱

**Files:**
- Modify: `transcribe.py`, `test_transcribe_model_cache.py`

Whisper처럼 pyannote Pipeline도 전역 변수로 캐싱해 동일 프로세스 내 재호출 시 모델 재로드를 방지한다.

- [ ] **Step 1: 실패 테스트 작성**

`test_transcribe_model_cache.py` 에 클래스 추가:

```python
class PyannoteModelCacheTest(unittest.TestCase):
    def test_get_pyannote_pipeline_reuses_single_instance(self):
        transcribe._PYANNOTE_PIPELINE = None

        with patch("transcribe.Pipeline") as pipeline_cls:
            pipeline_cls.from_pretrained.return_value = object()

            first = transcribe.get_pyannote_pipeline("fake-token")
            second = transcribe.get_pyannote_pipeline("fake-token")

        self.assertIs(first, second)
        self.assertEqual(pipeline_cls.from_pretrained.call_count, 1)
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
python -m pytest test_transcribe_model_cache.py::PyannoteModelCacheTest -v
```

Expected: `AttributeError: module 'transcribe' has no attribute 'get_pyannote_pipeline'`

- [ ] **Step 3: transcribe.py에 구현 추가**

`_WHISPER_MODEL = None` 바로 아래에 추가:

```python
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
```

`run_diarization()` 함수 내 `Pipeline.from_pretrained(...)` 직접 호출을 교체:

```python
def run_diarization(audio_path: str, num_speakers: Optional[int] = None) -> list[tuple]:
    """pyannote로 화자 분리. [(speaker, start, end), ...] 반환."""
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN 환경변수가 설정되지 않았습니다.")

    print("화자 분리 실행 중...")
    pipeline = get_pyannote_pipeline(hf_token)  # 캐싱된 pipeline 사용
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
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
python -m pytest test_transcribe_model_cache.py -v
```

Expected: 기존 Whisper 캐싱 테스트 + 신규 pyannote 캐싱 테스트 전체 PASS

- [ ] **Step 5: 커밋**

```bash
git add transcribe.py test_transcribe_model_cache.py
git commit -m "perf(transcribe): cache pyannote pipeline to avoid repeated model loading"
```

---

### Task 5: Whisper + pyannote 병렬 실행

**Files:**
- Modify: `transcribe.py`
- Create: `test_transcribe_parallel.py`

화자분리가 있을 때 STT와 화자분리를 `ThreadPoolExecutor`로 병렬 실행한다. 두 작업은 동일 파일을 읽기만 하므로 race condition 없음.

- [ ] **Step 1: 실패 테스트 작성**

`test_transcribe_parallel.py` 생성:

```python
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
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
python -m pytest test_transcribe_parallel.py -v
```

Expected: `test_whisper_and_diarization_both_called`은 현재 `transcribe()` 내부 구현이 `run_whisper`/`run_diarization`을 직접 호출하므로 PASS. `test_parallel_result_merges_correctly`도 동작 결과가 동일해 PASS. 이 단계의 목적은 구현 전 인터페이스 계약을 코드로 고정하는 것.

- [ ] **Step 3: transcribe.py `transcribe()` 함수 수정**

`transcribe.py` 상단 import에 추가:

```python
from concurrent.futures import ThreadPoolExecutor
```

기존 `transcribe()` 함수의 STT + 화자분리 부분 교체:

```python
def transcribe(
    audio_path: str,
    num_speakers: Optional[int] = None,
    initial_prompt: Optional[str] = None,
    skip_diarization: bool = False,
) -> list[dict]:
    """전체 파이프라인: STT + (선택적) 화자분리 병렬 실행 + 병합."""
    if skip_diarization:
        segments = run_whisper(audio_path, initial_prompt=initial_prompt)
        return [{"speaker": "Speaker A", "start": s["start"],
                 "end": s["end"], "text": s["text"]} for s in segments]

    # STT와 화자분리를 병렬로 실행 (두 작업은 독립적)
    with ThreadPoolExecutor(max_workers=2) as executor:
        stt_future = executor.submit(run_whisper, audio_path, initial_prompt)
        diar_future = executor.submit(run_diarization, audio_path, num_speakers)

    segments = stt_future.result()
    turns = diar_future.result()
    return merge(segments, turns)
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
python -m pytest test_transcribe_parallel.py -v
```

Expected: 3개 PASS

- [ ] **Step 5: 전체 transcribe 테스트 회귀 확인**

```bash
python -m pytest test_transcribe_model_cache.py test_transcribe_mlx.py test_transcribe_parallel.py -v
```

Expected: 전체 PASS

- [ ] **Step 6: 커밋**

```bash
git add transcribe.py test_transcribe_parallel.py
git commit -m "perf(transcribe): run STT and diarization in parallel with ThreadPoolExecutor"
```

---

## Chunk 3: pipeline.py VAD 개선 + requirements.txt

### Task 6: VAD 이중 read 제거

**Files:**
- Modify: `pipeline.py`
- Create: `test_pipeline_vad.py`

현재 `get_vad_speech_segments()`와 `run_vad()`가 각각 `read_audio()`를 호출한다. `get_vad_speech_segments()`를 제거하고 `run_vad()`에 통합해 wav를 한 번만 읽는다.

- [ ] **Step 1: 실패 테스트 작성**

`test_pipeline_vad.py` 생성:

```python
import unittest
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pipeline


class VadSingleReadTest(unittest.TestCase):
    def test_read_audio_called_once(self):
        """run_vad() 실행 시 read_audio가 정확히 1번만 호출되는지 확인."""
        fake_wav = MagicMock()
        fake_wav.__len__ = lambda s: 16000 * 10  # 10초

        fake_timestamps = [{"start": 0, "end": 8000}, {"start": 16000, "end": 32000}]
        fake_chunks = MagicMock()
        fake_chunks.__len__ = lambda s: 16000 * 8  # 8초

        with patch("pipeline.load_silero_vad") as load_fn, \
             patch("pipeline.read_audio", return_value=fake_wav) as read_fn, \
             patch("pipeline.get_speech_timestamps", return_value=fake_timestamps), \
             patch("pipeline.collect_chunks", return_value=fake_chunks), \
             patch("pipeline.save_audio"):
            load_fn.return_value = MagicMock()
            result = pipeline.run_vad(Path("/tmp/test.wav"))

        self.assertEqual(read_fn.call_count, 1)

    def test_run_vad_returns_original_on_no_speech(self):
        """발화 구간이 없으면 원본 파일 경로를 반환하는지 확인."""
        fake_wav = MagicMock()
        fake_wav.__len__ = lambda s: 16000 * 5

        with patch("pipeline.load_silero_vad") as load_fn, \
             patch("pipeline.read_audio", return_value=fake_wav), \
             patch("pipeline.get_speech_timestamps", return_value=[]):
            load_fn.return_value = MagicMock()
            original = Path("/tmp/audio.wav")
            result = pipeline.run_vad(original)

        self.assertEqual(result, original)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
python -m pytest test_pipeline_vad.py -v
```

Expected: `test_read_audio_called_once` FAIL (현재 2회 호출)

- [ ] **Step 3: pipeline.py 수정**

`get_vad_speech_segments()` 함수를 제거하고, `run_vad()`를 다음으로 교체:

```python
def run_vad(path: Path) -> Path:
    """silero-vad로 무음 구간 제거. VAD 처리된 wav를 원본과 같은 디렉토리에 저장."""
    from silero_vad import (
        collect_chunks, get_speech_timestamps, load_silero_vad,
        read_audio, save_audio,
    )

    # wav를 한 번만 읽어 VAD와 청크 추출에 모두 사용
    wav = read_audio(str(path), sampling_rate=16000)
    duration_sec = len(wav) / 16000

    model = load_silero_vad()
    speech_timestamps = get_speech_timestamps(
        wav, model,
        sampling_rate=16000,
        speech_pad_ms=300,
        min_silence_duration_ms=VAD_MIN_SILENCE_DURATION_MS,
    )

    if not speech_timestamps:
        print("VAD: 발화 구간 없음, 원본 사용")
        return path

    audio_chunks = collect_chunks(speech_timestamps, wav)
    out_path = path.parent / f"{path.stem}_vad.wav"
    save_audio(str(out_path), audio_chunks, sampling_rate=16000)

    vad_sec = len(audio_chunks) / 16000
    print(f"VAD 완료: {duration_sec:.1f}s → {vad_sec:.1f}s ({100*vad_sec/duration_sec:.0f}% 유지)")
    return out_path
```

`pipeline.run()` 에서 `get_vad_speech_segments()` 직접 호출이 없으니 추가 수정 불필요.

- [ ] **Step 4: 테스트 통과 확인**

```bash
python -m pytest test_pipeline_vad.py -v
```

Expected: 2개 PASS

- [ ] **Step 5: 기존 파이프라인 테스트 회귀 확인**

```bash
python -m pytest test_pipeline.py test_pipeline_vad.py -v
```

Expected: 전체 PASS

- [ ] **Step 6: 커밋**

```bash
git add pipeline.py test_pipeline_vad.py
git commit -m "perf(pipeline): eliminate duplicate read_audio call in run_vad"
```

---

### Task 7: requirements.txt 정리

**Files:**
- Modify: `requirements.txt`

`openai-whisper`는 코드에서 미사용. 제거 후 `mlx-whisper` 추가.

- [ ] **Step 1: requirements.txt 수정**

`openai-whisper>=20231117` 줄 제거.
`faster_whisper` 아래에 추가:

```
mlx-whisper>=0.4.0
```

최종 결과:

```
sounddevice==0.4.6
scipy>=1.11.0
faster-whisper>=1.0.0
mlx-whisper>=0.4.0
pyannote.audio==3.1.1
anthropic>=0.20.0
notion-client==2.2.1
python-dotenv==1.0.0
numpy>=1.24.0
pdfplumber>=0.10.0
python-docx>=1.1.0
openpyxl>=3.1.0
```

- [ ] **Step 2: 전체 테스트 최종 확인**

```bash
cd ~/meeting_tools && python -m pytest test_transcribe_model_cache.py test_transcribe_mlx.py test_transcribe_parallel.py test_pipeline.py test_pipeline_vad.py test_pipeline_correction.py test_context_loader.py -v
```

Expected: 전체 PASS

- [ ] **Step 3: 커밋**

```bash
git add requirements.txt
git commit -m "chore: remove unused openai-whisper, add mlx-whisper dependency"
```
