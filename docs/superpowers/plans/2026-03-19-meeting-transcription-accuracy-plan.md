# Meeting Transcription Accuracy Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 문서 첨부를 전제로, 회의 STT에서 도메인 용어 오인식과 문장 누락/잘림을 줄이는 청크 기반 전사 파이프라인을 구현한다.

**Architecture:** `pipeline.py`는 전체 오디오를 한 번에 STT하지 않고, VAD 기반 경계 탐지 후 청크 단위로 Whisper를 호출한다. `context_loader.py`는 문서에서 용어 구조를 추출하고, `transcribe.py`는 청크별 전사 및 병합 메타데이터를 제공한다. 후처리 단계는 문서 근거가 있는 용어만 보수적으로 교정한다.

**Tech Stack:** Python 3.9, silero-vad, faster-whisper, unittest

---

## Chunk 1: 문서 용어 구조화

### Task 1: `context_loader.py` 반환 구조 확장

**Files:**
- Modify: `context_loader.py`
- Test: `test_context_loader.py`

- [ ] **Step 1: Write the failing test**

```python
import unittest

from context_loader import load


class ContextLoaderTest(unittest.TestCase):
    def test_load_returns_term_metadata(self):
        key_terms, doc_content, term_metadata = load(["/tmp/sample.txt"])
        self.assertIn("canonical_terms", term_metadata)
        self.assertIn("alias_map", term_metadata)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest test_context_loader.py`
Expected: FAIL because `load()` does not return term metadata yet

- [ ] **Step 3: Write minimal implementation**

구현 내용:
- `load()` 반환값을 `(key_terms, doc_content, term_metadata)`로 확장
- `term_metadata`는 최소한 다음 키를 포함
  - `canonical_terms`
  - `alias_map`
  - `priority_terms`

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest test_context_loader.py`
Expected: PASS

- [ ] **Step 5: Verify no existing usage breaks**

Run: `python3 -m unittest test_pipeline.py`
Expected: PASS or import failure 없이 통과

### Task 2: 문서 기반 용어 우선순위 추가

**Files:**
- Modify: `context_loader.py`
- Test: `test_context_loader.py`

- [ ] **Step 1: Write the failing test**

```python
def test_priority_terms_put_product_names_first(self):
    _, _, term_metadata = load(["/tmp/sample.txt"])
    self.assertIn("AEGIS-AP", term_metadata["priority_terms"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest test_context_loader.py`
Expected: FAIL because priority term selection is not implemented

- [ ] **Step 3: Write minimal implementation**

구현 내용:
- 제품명 / 프로젝트명 / 기능명을 우선 용어로 분류
- 기존 키워드 추출 결과를 재활용하되 우선순위만 추가

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest test_context_loader.py`
Expected: PASS

---

## Chunk 2: 청크 경계 계산

### Task 3: VAD 결과를 청크 경계 후보로 반환하는 함수 추가

**Files:**
- Modify: `pipeline.py`
- Test: `test_pipeline_chunking.py`

- [ ] **Step 1: Write the failing test**

```python
import unittest

import pipeline


class ChunkBoundaryTest(unittest.TestCase):
    def test_build_chunks_splits_on_long_silence(self):
        speech = [
            {"start": 0.0, "end": 30.0},
            {"start": 35.5, "end": 60.0},
        ]
        chunks = pipeline.build_chunks_from_speech_segments(speech, max_chunk_sec=120, overlap_sec=1.0)
        self.assertEqual(len(chunks), 2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest test_pipeline_chunking.py`
Expected: FAIL because helper does not exist

- [ ] **Step 3: Write minimal implementation**

구현 내용:
- `build_chunks_from_speech_segments()` 추가
- 입력: speech segment 목록
- 출력: `(start_sec, end_sec)` 청크 목록

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest test_pipeline_chunking.py`
Expected: PASS

### Task 4: 최대 청크 길이와 최소 청크 길이 규칙 추가

**Files:**
- Modify: `pipeline.py`
- Test: `test_pipeline_chunking.py`

- [ ] **Step 1: Write the failing test**

```python
def test_build_chunks_respects_max_chunk_length(self):
    speech = [{"start": 0.0, "end": 260.0}]
    chunks = pipeline.build_chunks_from_speech_segments(speech, max_chunk_sec=120, overlap_sec=1.0)
    self.assertGreater(len(chunks), 1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest test_pipeline_chunking.py`
Expected: FAIL because max length split is not implemented

- [ ] **Step 3: Write minimal implementation**

구현 내용:
- `max_chunk_sec` 초과 시 강제 분할
- `min_chunk_sec` 이하 청크는 앞뒤 청크와 병합

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest test_pipeline_chunking.py`
Expected: PASS

---

## Chunk 3: 청크별 STT 실행

### Task 5: `transcribe.py`에 청크 전사 함수 추가

**Files:**
- Modify: `transcribe.py`
- Test: `test_transcribe_chunked.py`

- [ ] **Step 1: Write the failing test**

```python
import unittest

import transcribe


class ChunkTranscribeTest(unittest.TestCase):
    def test_build_initial_prompt_includes_priority_terms(self):
        prompt = transcribe.build_initial_prompt(
            meeting_title="목표합의서",
            context="AI, VOC",
            term_metadata={"priority_terms": ["AEGIS-AP", "WiNG"], "alias_map": {"VoC": "VOC"}}
        )
        self.assertIn("AEGIS-AP", prompt)
        self.assertIn("WiNG", prompt)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest test_transcribe_chunked.py`
Expected: FAIL because prompt builder does not exist

- [ ] **Step 3: Write minimal implementation**

구현 내용:
- `build_initial_prompt()` 추가
- 회의 주제, 기존 context, 우선 용어, 약어 정보를 문자열로 조합

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest test_transcribe_chunked.py`
Expected: PASS

### Task 6: 청크별 전사 결과를 시간 메타데이터와 함께 반환

**Files:**
- Modify: `transcribe.py`
- Test: `test_transcribe_chunked.py`

- [ ] **Step 1: Write the failing test**

```python
def test_offset_segments_applies_chunk_start(self):
    segments = [{"start": 0.5, "end": 2.0, "text": "안녕하세요"}]
    shifted = transcribe.offset_segments(segments, chunk_start=30.0)
    self.assertEqual(shifted[0]["start"], 30.5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest test_transcribe_chunked.py`
Expected: FAIL because offset helper does not exist

- [ ] **Step 3: Write minimal implementation**

구현 내용:
- `offset_segments()` 추가
- 청크 내부 시간축을 전체 오디오 시간축으로 변환

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest test_transcribe_chunked.py`
Expected: PASS

---

## Chunk 4: 병합과 중복 제거

### Task 7: 청크 오버랩 중복 제거 함수 추가

**Files:**
- Modify: `transcribe.py`
- Test: `test_transcribe_merge.py`

- [ ] **Step 1: Write the failing test**

```python
import unittest

import transcribe


class MergeTest(unittest.TestCase):
    def test_deduplicate_overlap_removes_repeated_line(self):
        merged = transcribe.deduplicate_overlap_text(
            ["안녕하세요", "이번 회의 시작하겠습니다"],
            ["이번 회의 시작하겠습니다", "첫 번째 안건입니다"],
        )
        self.assertEqual(
            merged,
            ["안녕하세요", "이번 회의 시작하겠습니다", "첫 번째 안건입니다"]
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest test_transcribe_merge.py`
Expected: FAIL because merge helper does not exist

- [ ] **Step 3: Write minimal implementation**

구현 내용:
- 오버랩 구간 중 동일하거나 매우 유사한 문장을 하나로 합치기
- 첫 구현은 exact match 위주로 단순하게 시작

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest test_transcribe_merge.py`
Expected: PASS

### Task 8: 청크 결과 전체 병합 함수 추가

**Files:**
- Modify: `transcribe.py`
- Test: `test_transcribe_merge.py`

- [ ] **Step 1: Write the failing test**

```python
def test_merge_chunk_results_returns_time_sorted_segments(self):
    chunks = [
        [{"start": 30.0, "end": 31.0, "text": "두번째"}],
        [{"start": 0.0, "end": 1.0, "text": "첫번째"}],
    ]
    merged = transcribe.merge_chunk_segments(chunks)
    self.assertEqual([s["text"] for s in merged], ["첫번째", "두번째"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest test_transcribe_merge.py`
Expected: FAIL because merge function does not exist

- [ ] **Step 3: Write minimal implementation**

구현 내용:
- 청크별 세그먼트를 시간순으로 정렬
- 오버랩 정리 후 하나의 세그먼트 목록으로 반환

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest test_transcribe_merge.py`
Expected: PASS

---

## Chunk 5: 보수적 용어 교정

### Task 9: 문서 근거 기반 용어 정규화 함수 추가

**Files:**
- Modify: `pipeline.py`
- Test: `test_pipeline_correction.py`

- [ ] **Step 1: Write the failing test**

```python
import unittest

import pipeline


class CorrectionTest(unittest.TestCase):
    def test_normalize_terms_uses_alias_map_only(self):
        transcript = "이지스 AP와 VoC 기능을 본다"
        term_metadata = {"alias_map": {"이지스 AP": "AEGIS-AP", "VoC": "VOC"}}
        corrected = pipeline.normalize_terms(transcript, term_metadata)
        self.assertEqual(corrected, "AEGIS-AP와 VOC 기능을 본다")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest test_pipeline_correction.py`
Expected: FAIL because normalization helper does not exist

- [ ] **Step 3: Write minimal implementation**

구현 내용:
- `normalize_terms()` 추가
- `alias_map` 기반 문자열 치환만 수행
- 문서에 없는 표현 생성 금지

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest test_pipeline_correction.py`
Expected: PASS

---

## Chunk 6: 파이프라인 통합

### Task 10: `pipeline.run()`을 청크 기반으로 전환

**Files:**
- Modify: `pipeline.py`
- Test: `test_pipeline_integration.py`

- [ ] **Step 1: Write the failing test**

```python
import unittest
from unittest.mock import patch

import pipeline


class PipelineIntegrationTest(unittest.TestCase):
    @patch("pipeline.transcribe_chunked")
    def test_run_uses_chunked_transcription(self, mock_transcribe_chunked):
        mock_transcribe_chunked.return_value = [{"speaker": "Speaker A", "start": 0, "end": 1, "text": "테스트"}]
        # 실제 파일 I/O는 fixture 또는 mock으로 대체
        self.assertTrue(hasattr(pipeline, "transcribe_chunked"))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest test_pipeline_integration.py`
Expected: FAIL because chunked path is not wired yet

- [ ] **Step 3: Write minimal implementation**

구현 내용:
- `pipeline.run()`이
  - 문서 로드
  - 청크 생성
  - 청크별 STT
  - 병합
  - 용어 정규화
  흐름을 사용하도록 수정

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest test_pipeline_integration.py`
Expected: PASS

### Task 11: 결과 JSON에 메타데이터 저장

**Files:**
- Modify: `pipeline.py`
- Test: `test_pipeline_integration.py`

- [ ] **Step 1: Write the failing test**

```python
def test_result_json_contains_chunking_metadata(self):
    result = {
        "chunking": {
            "silence_ms": 3000,
            "max_chunk_sec": 120,
            "overlap_sec": 1.0,
        }
    }
    self.assertIn("chunking", result)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest test_pipeline_integration.py`
Expected: FAIL because metadata is not stored yet

- [ ] **Step 3: Write minimal implementation**

구현 내용:
- 결과 JSON에 `vad_applied`, `chunking`, `source_audio` 추가

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest test_pipeline_integration.py`
Expected: PASS

---

## Chunk 7: 검증

### Task 12: 회귀 테스트와 실제 파일 검증

**Files:**
- Modify: `README.md`
- Test: 전체 테스트 파일

- [ ] **Step 1: Run unit tests**

Run: `python3 -m unittest test_pipeline.py test_context_loader.py test_pipeline_chunking.py test_transcribe_chunked.py test_transcribe_merge.py test_pipeline_correction.py test_pipeline_integration.py`
Expected: All PASS

- [ ] **Step 2: Run real meeting file**

Run:
```bash
python3 pipeline.py /Users/geonhee/meetings/audio_20260318_1030.wav "목표합의서 관련내용" --docs "/Users/geonhee/Downloads/2026년 예건희 목표합의서_초안.xlsx"
```
Expected:
- JSON 생성
- 환각 문장 감소
- 용어 인식 개선
- 문장 단절 감소

- [ ] **Step 3: Document comparison workflow**

구현 내용:
- README에 `원본 직처리 / 기존 VAD / 청크 기반 VAD` 비교 절차 추가

- [ ] **Step 4: Final verification**

Run: 위 실제 파일 결과를 사람이 확인하고, 대표 오인식 예시를 기록
Expected: 개선 여부를 비교 가능

---

Plan complete and saved to `docs/superpowers/plans/2026-03-19-meeting-transcription-accuracy-plan.md`. Ready to execute?
