# Meeting STT Accuracy And VAD Policy Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `/meeting` 파이프라인을 원본 오디오 중심으로 재구성하고, VAD를 옵션화하면서 청크 기반 STT와 문서 기반 용어 강화를 통해 누락과 도메인 용어 오인식을 줄인다.

**Architecture:** `pipeline.py`가 실행 정책을 결정하고, `transcribe.py`가 청크 기반 STT와 프롬프트 구성을 담당한다. `context_loader.py`는 문서 기반 용어 근거를 제공하고, 스킬 레이어는 새 기본 정책을 사용자 인터페이스에 반영한다.

**Tech Stack:** Python 3, pytest, faster-whisper/mlx-whisper, silero-vad, existing meeting skill orchestration

---

## 파일 구조

- Modify: `/Users/geonhee/meeting_tools/pipeline.py`
  - 원본 오디오 기본 정책, VAD 옵션화, 결과 메타데이터 저장
- Modify: `/Users/geonhee/meeting_tools/transcribe.py`
  - 청크 기반 STT, 프롬프트 구성, 오버랩 병합 강화
- Modify: `/Users/geonhee/meeting_tools/test_pipeline.py`
  - 기본 경로와 메타데이터 테스트 추가
- Modify: `/Users/geonhee/meeting_tools/test_pipeline_vad.py`
  - VAD 옵션 및 fallback 테스트 보강
- Modify: `/Users/geonhee/meeting_tools/test_transcribe_parallel.py`
  - 청크 병합/중복 제거 테스트 보강
- Modify: `/Users/geonhee/meeting_tools/test_context_loader.py`
  - 용어 메타데이터 활용 검증 보강
- Modify: `/Users/geonhee/dotfiles/skills/meeting/SKILL.md`
  - 기본 정책과 옵션 설명 갱신

## Chunk 1: Default Policy and Metadata

### Task 1: 기본 경로를 원본 오디오 기반으로 고정

**Files:**
- Modify: `/Users/geonhee/meeting_tools/test_pipeline.py`
- Modify: `/Users/geonhee/meeting_tools/pipeline.py`
- Test: `/Users/geonhee/meeting_tools/test_pipeline.py`

- [ ] **Step 1: 기본 원본 오디오 경로 테스트 작성**

```python
def test_run_uses_original_audio_by_default(...):
    ...
    assert captured_audio_path == expected_original_path
    assert result["vad_applied"] is False
```

- [ ] **Step 2: 테스트 실행**

Run: `cd /Users/geonhee/meeting_tools && pytest test_pipeline.py -k original_audio -v`
Expected: FAIL because VAD still runs by default

- [ ] **Step 3: `pipeline.py`에 `use_vad=False` 기본 정책 반영**

```python
def run(..., use_vad: bool = False) -> str:
    stt_input_path = path
```

- [ ] **Step 4: 결과 JSON 메타데이터 추가**

```python
result = {
    ...
    "vad_applied": vad_applied,
    "chunking_applied": True,
    "source_audio_path": str(original_path),
    "stt_audio_path": str(stt_input_path),
}
```

- [ ] **Step 5: 테스트 재실행**

Run: `cd /Users/geonhee/meeting_tools && pytest test_pipeline.py -k original_audio -v`
Expected: PASS

## Chunk 2: Optional VAD and Fallback

### Task 2: VAD 옵션과 fallback 정책 구현

**Files:**
- Modify: `/Users/geonhee/meeting_tools/test_pipeline_vad.py`
- Modify: `/Users/geonhee/meeting_tools/pipeline.py`
- Test: `/Users/geonhee/meeting_tools/test_pipeline_vad.py`

- [ ] **Step 1: VAD 옵션 테스트 작성**

```python
def test_run_uses_vad_audio_only_when_requested(...):
    ...
    assert captured_audio_path == expected_vad_path
    assert result["vad_applied"] is True
```

- [ ] **Step 2: VAD fallback 테스트 작성**

```python
def test_run_falls_back_to_original_audio_when_vad_fails(...):
    ...
    assert captured_audio_path == expected_original_path
    assert result["vad_applied"] is False
```

- [ ] **Step 3: 테스트 실행**

Run: `cd /Users/geonhee/meeting_tools && pytest test_pipeline_vad.py -v`
Expected: FAIL before fallback implementation

- [ ] **Step 4: `pipeline.py`에 VAD fallback 구현**

```python
if use_vad:
    try:
        stt_input_path = run_vad(path)
        vad_applied = True
    except Exception:
        print("VAD 실패, 원본 오디오로 계속 진행")
        stt_input_path = path
        vad_applied = False
```

- [ ] **Step 5: 테스트 재실행**

Run: `cd /Users/geonhee/meeting_tools && pytest test_pipeline_vad.py -v`
Expected: PASS

## Chunk 3: Chunked STT

### Task 3: 청크 기반 STT와 오버랩 병합 강화

**Files:**
- Modify: `/Users/geonhee/meeting_tools/transcribe.py`
- Modify: `/Users/geonhee/meeting_tools/test_transcribe_parallel.py`
- Test: `/Users/geonhee/meeting_tools/test_transcribe_parallel.py`

- [ ] **Step 1: 청크 병합 관련 테스트 작성**

```python
def test_merge_chunk_segments_removes_overlap_duplicates():
    ...
```

- [ ] **Step 2: 프롬프트/청크 처리 테스트 작성**

```python
def test_build_initial_prompt_includes_priority_terms_and_aliases():
    ...
```

- [ ] **Step 3: 테스트 실행**

Run: `cd /Users/geonhee/meeting_tools && pytest test_transcribe_parallel.py -v`
Expected: FAIL or reveal missing behaviors

- [ ] **Step 4: `transcribe.py`에 청크 기반 기본 경로 보강**

```python
# 긴 입력을 청크 단위로 처리하고 병합한다.
# 오버랩 구간의 중복 문장은 제거한다.
```

- [ ] **Step 5: 프롬프트 구조 개선**

```python
parts = [
    f"회의 주제: {meeting_title}",
    f"핵심 용어: ...",
    f"표기 참고: alias -> canonical",
]
```

- [ ] **Step 6: 테스트 재실행**

Run: `cd /Users/geonhee/meeting_tools && pytest test_transcribe_parallel.py -v`
Expected: PASS

## Chunk 4: Term Accuracy

### Task 4: 문서 기반 용어 반영과 정규화 검증

**Files:**
- Modify: `/Users/geonhee/meeting_tools/test_context_loader.py`
- Modify: `/Users/geonhee/meeting_tools/pipeline.py`
- Test: `/Users/geonhee/meeting_tools/test_context_loader.py`

- [ ] **Step 1: 용어 메타데이터 반영 테스트 작성**

```python
def test_normalize_terms_uses_alias_map_only_when_documented():
    ...
```

- [ ] **Step 2: 문서 로드 실패 시 기본 STT 진행 테스트 작성**

```python
def test_pipeline_continues_without_term_boost_when_doc_loading_fails():
    ...
```

- [ ] **Step 3: 테스트 실행**

Run: `cd /Users/geonhee/meeting_tools && pytest test_context_loader.py -v`
Expected: FAIL if current handling is insufficient

- [ ] **Step 4: 최소 구현 보강**

```python
try:
    key_terms, doc_content, term_metadata, agenda_items = load_context(doc_paths)
except Exception:
    key_terms, doc_content, term_metadata, agenda_items = "", "", {}, []
```

- [ ] **Step 5: 테스트 재실행**

Run: `cd /Users/geonhee/meeting_tools && pytest test_context_loader.py -v`
Expected: PASS

## Chunk 5: Skill Interface and Verification

### Task 5: `/meeting` 스킬 정책과 실행 예시 정렬

**Files:**
- Modify: `/Users/geonhee/dotfiles/skills/meeting/SKILL.md`
- Test: `/Users/geonhee/dotfiles/skills/meeting/SKILL.md`

- [ ] **Step 1: 기본 정책 설명 수정**

```md
- 기본 실행은 원본 오디오 + 청크 기반 STT
- VAD는 명시 옵션일 때만 적용
```

- [ ] **Step 2: 실행 명령 예시 갱신**

```bash
python3 pipeline.py <오디오파일경로> [회의제목] [--vad] [--speakers N] [--context "..."] [--docs ...]
```

- [ ] **Step 3: 문서 검토**

Run: `sed -n '1,220p' /Users/geonhee/dotfiles/skills/meeting/SKILL.md`
Expected: 기본 정책과 옵션 정책이 일관되게 보인다

### Task 6: 전체 검증

**Files:**
- Test: `/Users/geonhee/meeting_tools/test_pipeline.py`
- Test: `/Users/geonhee/meeting_tools/test_pipeline_vad.py`
- Test: `/Users/geonhee/meeting_tools/test_transcribe_parallel.py`
- Test: `/Users/geonhee/meeting_tools/test_context_loader.py`

- [ ] **Step 1: 관련 테스트 전체 실행**

Run: `cd /Users/geonhee/meeting_tools && pytest test_pipeline.py test_pipeline_vad.py test_transcribe_parallel.py test_context_loader.py -v`
Expected: PASS

- [ ] **Step 2: 샘플 파일 수동 비교**

Run:

```bash
cd /Users/geonhee/meeting_tools && python3 pipeline.py ~/meetings/sample.wav "정확도 비교"
cd /Users/geonhee/meeting_tools && python3 pipeline.py ~/meetings/sample.wav "정확도 비교" --vad
```

Expected:
- 기본 경로는 원본 오디오 기반
- VAD 경로는 옵션일 때만 적용
- 새 기본 경로가 기존 방식보다 누락/잘림이 적다
- 도메인 용어 표기가 더 안정적이다
