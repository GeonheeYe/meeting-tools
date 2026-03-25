# Meeting Audio Enhancement Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 원본 오디오는 그대로 보존하면서, STT 직전에만 사용할 보정본 오디오를 만들어 작은 목소리 인식률을 높인다.

**Architecture:** `pipeline.py`에 오디오 보정 단계를 추가한다. 보정본은 임시 wav로 생성하고, 이후 VAD/STT는 그 파일을 기준으로 동작한다. 보정 실패 시 원본 경로로 fallback한다.

**Tech Stack:** Python 3, ffmpeg, pytest

---

## 파일 구조

- Modify: `/Users/geonhee/meeting_tools/pipeline.py`
  - STT용 오디오 보정 단계 추가
- Modify: `/Users/geonhee/meeting_tools/test_pipeline.py`
  - `audio_enhanced` 메타데이터 테스트 추가
- Create: `/Users/geonhee/meeting_tools/test_pipeline_audio_enhancement.py`
  - 보정 함수 및 fallback 테스트

## Chunk 1: Audio Enhancement TDD

### Task 1: 보정본 생성과 fallback을 테스트로 고정

**Files:**
- Modify: `/Users/geonhee/meeting_tools/test_pipeline.py`
- Create: `/Users/geonhee/meeting_tools/test_pipeline_audio_enhancement.py`
- Test: `/Users/geonhee/meeting_tools/test_pipeline_audio_enhancement.py`

- [ ] **Step 1: 보정본 생성 테스트 작성**

```python
def test_enhance_audio_for_stt_creates_temp_file():
    ...
```

- [ ] **Step 2: fallback 테스트 작성**

```python
def test_pipeline_falls_back_when_audio_enhancement_fails():
    ...
```

- [ ] **Step 3: JSON 메타데이터 테스트 작성**

```python
def test_result_json_marks_audio_enhanced():
    ...
```

- [ ] **Step 4: 테스트 실행**

Run: `cd /Users/geonhee/meeting_tools && python3 -m pytest test_pipeline.py test_pipeline_audio_enhancement.py -v`
Expected: FAIL before implementation

- [ ] **Step 5: `pipeline.py` 구현**

```python
def enhance_audio_for_stt(path: Path) -> Path:
    ...
```

- [ ] **Step 6: 테스트 재실행**

Run: `cd /Users/geonhee/meeting_tools && python3 -m pytest test_pipeline.py test_pipeline_audio_enhancement.py -v`
Expected: PASS

## Chunk 2: Manual Verification

### Task 2: 실제 회의 파일로 품질 확인

**Files:**
- Test: `/Users/geonhee/meetings/audio_20260324_1259.wav`

- [ ] **Step 1: 보정본 경로로 파이프라인 실행**

Run:

```bash
cd /Users/geonhee/meeting_tools && python3 pipeline.py /Users/geonhee/meetings/audio_20260324_1259.wav --vad
```

- [ ] **Step 2: 원본과 보정본 비교 청취**

Check:
- 작은 목소리 구간이 더 잘 들리는지
- 큰 목소리 왜곡이 심하지 않은지

- [ ] **Step 3: STT 결과 비교**

Check:
- 중간 씹힘 감소 여부
- 반복 노이즈 변화 여부

- [ ] **Step 4: 결과 메모**

```text
- 작은 목소리 개선 체감
- 왜곡 발생 여부
- 추가 튜닝 필요 여부
```
