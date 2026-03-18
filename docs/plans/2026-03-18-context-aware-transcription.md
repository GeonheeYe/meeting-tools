# Context-Aware Transcription Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 참고 문서를 STT 파이프라인에 주입해서 도메인 용어 인식률을 높이고, Claude가 대화록을 교정+요약한다.

**Architecture:** context_loader가 문서에서 키워드를 추출해 Whisper initial_prompt를 풍부하게 만들고, pipeline이 문서 본문을 JSON에 저장한다. Claude(스킬 레이어)가 대화록+문서를 함께 읽어 STT 오류 교정 후 요약한다.

**Tech Stack:** Python 3.9+, pdfplumber (PDF), python-docx (DOCX), faster-whisper, pyannote (선택적)

---

## Chunk 1: context_loader.py

### Task 1: context_loader.py — TXT/MD 텍스트 추출 + 키워드 추출

**Files:**
- Create: `~/meeting_tools/context_loader.py`

- [ ] **Step 1: 테스트 파일 준비**

```bash
echo "XQBot은 schema linking을 제거한 Text-to-SQL 프레임워크입니다.
BIRD 벤치마크, Spider 1.0, GRPO 강화학습, pyannote 화자분리를 활용합니다." \
  > /tmp/test_context.txt
```

- [ ] **Step 2: context_loader.py 작성**

```python
# ~/meeting_tools/context_loader.py
"""
참고 문서에서 키워드와 본문을 추출한다.
- key_terms: Whisper initial_prompt에 삽입할 쉼표 구분 핵심 용어
- doc_content: Claude 교정 단계에 전달할 문서 본문
"""
import re
from pathlib import Path
from typing import Optional


def extract_text(path: Path) -> str:
    """파일 포맷에 맞게 텍스트 추출."""
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8")
    elif suffix == ".pdf":
        return _extract_pdf(path)
    elif suffix == ".docx":
        return _extract_docx(path)
    else:
        # 지원하지 않는 포맷은 빈 문자열
        print(f"[context_loader] 지원하지 않는 포맷: {suffix}, 건너뜀")
        return ""


def _extract_pdf(path: Path) -> str:
    import pdfplumber
    pages = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
    return "\n".join(pages)


def _extract_docx(path: Path) -> str:
    from docx import Document
    doc = Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def extract_key_terms(text: str, max_terms: int = 30) -> str:
    """
    텍스트에서 핵심 용어 추출 (규칙 기반).
    - 영문 대문자로 시작하는 단어/약어 (BIRD, XQBot, GRPO 등)
    - 한글 명사구 (2자 이상 연속 한글)
    - 중복 제거, max_terms 개까지
    """
    # 영문: 대문자 시작 단어 또는 전체 대문자 약어
    en_terms = re.findall(r'\b[A-Z][a-zA-Z0-9\-]+\b|\b[A-Z]{2,}\b', text)
    # 한글: 2자 이상 한글 명사구
    ko_terms = re.findall(r'[가-힣]{2,}', text)

    # 빈도 기반 정렬 (많이 등장한 용어 우선)
    from collections import Counter
    all_terms = en_terms + ko_terms
    counted = Counter(all_terms)

    # 너무 짧거나 일반적인 단어 제외
    _STOPWORDS = {"있다", "없다", "하다", "된다", "이다", "합니다", "입니다",
                  "있는", "없는", "하는", "되는", "이런", "그런", "저런",
                  "The", "This", "That", "For", "With", "And", "Are"}
    terms = [t for t, _ in counted.most_common(max_terms * 2)
             if t not in _STOPWORDS and len(t) >= 2]

    return ", ".join(terms[:max_terms])


def load(doc_paths: list[str]) -> tuple[str, str]:
    """
    문서 목록을 로드해서 (key_terms, doc_content) 반환.
    - key_terms: initial_prompt에 삽입
    - doc_content: Claude 교정용 전문
    """
    texts = []
    for p in doc_paths:
        path = Path(p).expanduser().resolve()
        if not path.exists():
            print(f"[context_loader] 파일 없음: {path}, 건너뜀")
            continue
        text = extract_text(path)
        if text:
            texts.append(f"=== {path.name} ===\n{text}")

    if not texts:
        return "", ""

    doc_content = "\n\n".join(texts)
    key_terms = extract_key_terms(doc_content)
    print(f"[context_loader] 키워드 추출 완료: {key_terms[:80]}...")
    return key_terms, doc_content


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("사용법: python context_loader.py <파일1> [파일2] ...")
        sys.exit(1)
    terms, content = load(sys.argv[1:])
    print(f"\n=== 키워드 ===\n{terms}")
    print(f"\n=== 본문 (앞 500자) ===\n{content[:500]}")
```

- [ ] **Step 3: 동작 확인**

```bash
cd ~/meeting_tools && python context_loader.py /tmp/test_context.txt
```

기대 출력:
```
[context_loader] 키워드 추출 완료: XQBot, Text, SQL, BIRD, Spider, GRPO, ...
=== 키워드 ===
XQBot, Text, SQL, BIRD, Spider, GRPO, ...
```

- [ ] **Step 4: 빈 파일 / 없는 파일 예외 처리 확인**

```bash
python context_loader.py /tmp/없는파일.txt
```

기대: "파일 없음" 경고 출력 후 `("", "")` 반환, 예외 없음

---

### Task 2: requirements.txt — pdfplumber, python-docx 추가

**Files:**
- Modify: `~/meeting_tools/requirements.txt`

- [ ] **Step 1: 의존성 추가**

`requirements.txt` 끝에 추가:
```
pdfplumber>=0.10.0
python-docx>=1.1.0
```

- [ ] **Step 2: 설치 확인**

```bash
cd ~/meeting_tools && pip install pdfplumber python-docx
```

기대: 에러 없이 설치 완료

- [ ] **Step 3: PDF 동작 확인 (PDF 파일이 있을 경우)**

```bash
# 테스트용 간단한 PDF가 있다면:
python context_loader.py ~/meetings/some_doc.pdf
```

---

## Chunk 2: pipeline.py 수정

### Task 3: pipeline.py — doc_paths 파라미터 + diarization 조건부

**Files:**
- Modify: `~/meeting_tools/pipeline.py`

현재 시그니처:
```python
def run(audio_path: str, title: Optional[str] = None,
        num_speakers: Optional[int] = None, context: Optional[str] = None) -> str:
```

- [ ] **Step 1: import 추가 및 시그니처 변경**

`pipeline.py` 상단 import에 추가:
```python
from context_loader import load as load_context
```

`run()` 시그니처 변경:
```python
def run(
    audio_path: str,
    title: Optional[str] = None,
    num_speakers: Optional[int] = None,
    context: Optional[str] = None,
    doc_paths: Optional[list] = None,
) -> str:
```

- [ ] **Step 2: context_loader 연동 블록 추가**

VAD 전처리 직전(`# VAD 전처리` 주석 위)에 삽입:

```python
# 참고 문서 로드 → initial_prompt 강화
doc_content = ""
if doc_paths:
    print(f"참고 문서 로드 중: {len(doc_paths)}개")
    key_terms, doc_content = load_context(doc_paths)
    if key_terms:
        context = f"{key_terms}, {context}" if context else key_terms
```

- [ ] **Step 3: diarization 조건부 처리**

현재 `transcribe()` 호출:
```python
merged = transcribe(str(vad_path), num_speakers=num_speakers, initial_prompt=context)
```

`transcribe.py`의 `run_diarization`은 이미 `num_speakers`가 없어도 동작하지만,
완전히 skip하려면 `transcribe.py`의 `transcribe()` 함수를 수정해야 한다.

`transcribe.py`의 `transcribe()` 함수 수정:

```python
def transcribe(
    audio_path: str,
    num_speakers: Optional[int] = None,
    initial_prompt: Optional[str] = None,
    skip_diarization: bool = False,
) -> list[dict]:
    """전체 파이프라인: STT + (선택적) 화자분리 + 병합."""
    segments = run_whisper(audio_path, initial_prompt=initial_prompt)

    if skip_diarization:
        # 화자 분리 없이 Speaker A 단일 레이블
        return [{"speaker": "Speaker A", "start": s["start"],
                 "end": s["end"], "text": s["text"]} for s in segments]

    turns = run_diarization(audio_path, num_speakers=num_speakers)
    return merge(segments, turns)
```

`pipeline.py`의 `transcribe()` 호출 수정:

```python
skip_diarization = num_speakers is None
merged = transcribe(
    str(vad_path),
    num_speakers=num_speakers,
    initial_prompt=context,
    skip_diarization=skip_diarization,
)
```

- [ ] **Step 4: JSON에 doc_content 저장**

`result` dict 구성 부분에 추가:

```python
result = {
    "title": title,
    "date": ts,
    "speaker_count": speaker_count,
    "transcript": transcript,
    "doc_content": doc_content,   # 추가 — Claude 교정용, 없으면 ""
}
```

- [ ] **Step 5: argparse에 --docs 옵션 추가**

```python
parser.add_argument("--docs", nargs="*", default=None,
                    help="참고 문서 경로 목록 (PDF, TXT, DOCX)")
```

`run()` 호출 수정:
```python
print(run(args.audio, args.title, args.speakers, args.context, args.docs))
```

- [ ] **Step 6: 동작 확인 (문서 없는 기존 방식)**

```bash
cd ~/meeting_tools && python pipeline.py --help
```

기대: `--docs` 옵션 포함된 help 출력

- [ ] **Step 7: 동작 확인 (화자 분리 없이 빠른 처리)**

```bash
# speakers 없이 실행하면 pyannote 생략 확인
python pipeline.py /tmp/test.wav "테스트" 2>&1 | grep -E "화자|VAD|STT"
```

기대: "화자 분리 실행 중..." 로그 없음

---

## Chunk 3: SKILL.md 수정

### Task 4: SKILL.md — 문서 파일 파싱 + Claude 교정+요약 통합

**Files:**
- Modify: `~/.claude/skills/meeting/SKILL.md`

- [ ] **Step 1: Step 1 파싱 규칙에 문서 파일 인식 추가**

현재 파싱 항목:
```
- 오디오 파일 경로
- 회의 제목
- 화자 수
- 컨텍스트
```

추가:
```
- **참고 문서**: `.pdf`, `.txt`, `.md`, `.docx` 확장자를 가진 경로들
```

예시 추가:
```
audio.wav XQBot 리뷰 2명 BIRD,GRPO agenda.txt notes.pdf
→ title=XQBot 리뷰, speakers=2, context=BIRD,GRPO, docs=[agenda.txt, notes.pdf]
```

- [ ] **Step 2: Step 2 pipeline 실행 명령에 --docs 옵션 추가**

```bash
cd ~/meeting_tools && python3 pipeline.py <오디오> [제목] [--speakers N] [--context "..."] [--docs doc1.pdf doc2.txt]
```

- [ ] **Step 3: Step 3 JSON 구조 업데이트**

```json
{
  "title": "...",
  "date": "...",
  "speaker_count": 2,
  "transcript": "...",
  "doc_content": "참고 문서 본문 (없으면 빈 문자열)"
}
```

- [ ] **Step 4: Step 4 Claude 교정+요약 통합으로 변경**

현재 Step 4:
```
transcript 내용을 바탕으로 요약/액션 아이템/주요 결정사항 작성
```

변경:
```
### Step 4: Claude 교정 + 요약

doc_content가 있으면 먼저 STT 오류 교정을 수행한다.

**교정 기준:**
- doc_content에 등장하는 고유명사/기술 용어가 transcript에서 다르게 표기된 경우 교정
- 예: "스키마 리킹" → "스키마 링킹" (doc에 "스키마 링킹" 있을 때)
- 추측으로 교정하지 말 것 — doc_content에 근거 있는 것만

교정 후 (또는 doc_content 없으면 바로):
- 회의 요약 (3-5줄)
- 액션 아이템 (`- [ ] 담당자: 내용 (기한)`)
- 주요 결정사항

교정된 transcript를 Notion 페이지 본문에 사용한다.
```

- [ ] **Step 5: 사용 예시 업데이트**

```
/meeting record
/meeting ~/meetings/audio.wav
/meeting ~/meetings/audio.wav 스프린트 회의 4명 RAG, 파이프라인
/meeting ~/meetings/audio.wav XQBot 리뷰 2명 BIRD,GRPO agenda.txt notes.pdf
```

---

## 완료 기준 체크리스트

- [ ] `python context_loader.py doc.txt` 실행 시 키워드 출력
- [ ] `python pipeline.py audio.wav "제목"` (speakers 없음) → pyannote 생략, 빠르게 완료
- [ ] `python pipeline.py audio.wav "제목" --speakers 2` → pyannote 실행
- [ ] `python pipeline.py audio.wav "제목" --docs doc.txt` → JSON에 `doc_content` 포함
- [ ] `/meeting audio.wav 제목 doc.txt` → Claude가 교정 후 요약
