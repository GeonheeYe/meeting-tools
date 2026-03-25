# Meeting Conservative VAD Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `/meeting`의 VAD를 유지하되, 실제 발화를 거의 자르지 않도록 훨씬 보수적인 무음 제거 파라미터로 조정한다.

**Architecture:** `pipeline.py`의 `run_vad()`는 그대로 유지하고, `min_silence_duration_ms`와 `speech_pad_ms`만 더 보수적으로 바꾼다. 검증은 자동 테스트와 샘플 오디오 청취를 함께 사용한다.

**Tech Stack:** Python 3, pytest, silero-vad

---

## 파일 구조

- Modify: `/Users/geonhee/meeting_tools/pipeline.py`
  - 보수적 VAD 파라미터 적용
- Modify: `/Users/geonhee/meeting_tools/test_pipeline.py`
  - VAD 침묵 기준 상수 검증 갱신
- Modify: `/Users/geonhee/meeting_tools/test_pipeline_vad.py`
  - `speech_pad_ms` 호출 값 검증 추가

## Chunk 1: Conservative VAD Parameters

### Task 1: 새 VAD 파라미터를 테스트로 고정

**Files:**
- Modify: `/Users/geonhee/meeting_tools/test_pipeline.py`
- Modify: `/Users/geonhee/meeting_tools/test_pipeline_vad.py`
- Test: `/Users/geonhee/meeting_tools/test_pipeline.py`
- Test: `/Users/geonhee/meeting_tools/test_pipeline_vad.py`

- [ ] **Step 1: 침묵 기준 테스트 수정**

```python
def test_vad_uses_more_conservative_silence_threshold():
    assert pipeline.VAD_MIN_SILENCE_DURATION_MS == 5000
```

- [ ] **Step 2: speech pad 테스트 추가**

```python
def test_run_vad_uses_larger_speech_pad():
    ...
    get_speech_timestamps.assert_called_once_with(
        ...,
        speech_pad_ms=1000,
        ...
    )
```

- [ ] **Step 3: 테스트 실행**

Run: `cd /Users/geonhee/meeting_tools && python3 -m pytest test_pipeline.py test_pipeline_vad.py -v`
Expected: FAIL before production code update

- [ ] **Step 4: `pipeline.py` 파라미터 수정**

```python
VAD_MIN_SILENCE_DURATION_MS = 5000
...
speech_pad_ms=1000
```

- [ ] **Step 5: 테스트 재실행**

Run: `cd /Users/geonhee/meeting_tools && python3 -m pytest test_pipeline.py test_pipeline_vad.py -v`
Expected: PASS

## Chunk 2: Manual Audio Verification

### Task 2: 실제 회의 파일로 수동 검증

**Files:**
- Test: `/Users/geonhee/meetings/audio_20260324_1259.wav`

- [ ] **Step 1: VAD 결과 생성**

Run:

```bash
cd /Users/geonhee/meeting_tools && python3 pipeline.py /Users/geonhee/meetings/audio_20260324_1259.wav --vad --docs /Users/geonhee/Downloads/2026-03-24.txt
```

- [ ] **Step 2: 원본과 VAD 결과 비교 청취**

Check:
- 새 `*_vad.wav`가 기존보다 발화를 덜 잘라먹는지
- 짧은 멈춤이 유지되는지

- [ ] **Step 3: STT 결과 확인**

Check:
- 문장 잘림이 줄었는지
- 반복 노이즈가 과도하게 늘지 않았는지

- [ ] **Step 4: 결과 메모**

```text
- 기존 VAD 대비 발화 손실 감소 여부
- 처리 시간 체감
- 추가 튜닝 필요 여부
```
