# Agenda-Aware Action Items Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 안건 파일(txt)이 제공될 때 액션 아이템을 안건 항목별로 그룹화해서 출력한다.

**Architecture:** `context_loader.py`가 안건 항목을 추출해 `load()` 4-tuple로 반환 → `pipeline.py`가 result JSON에 포함 → `/meeting` 스킬 Step 4가 항목별 그룹화 출력. 파이프라인 흐름과 비변경 파일(transcribe, record, summarize, notion_upload)은 그대로 유지.

**Tech Stack:** Python 3.9+, re (stdlib), pytest, SKILL.md (meeting skill)

---

## Chunk 1: context_loader.py — extract_agenda_items + load() 변경

### Task 1: `extract_agenda_items()` 함수 추가

**Files:**
- Modify: `context_loader.py`
- Test: `test_context_loader.py`

- [ ] **Step 1: 실패 테스트 작성**

`test_context_loader.py` 끝에 추가:

```python
class ExtractAgendaItemsTest(unittest.TestCase):
    def test_numbered_items_extracted(self):
        from context_loader import extract_agenda_items
        text = (
            "1. VQML: ITU-T 표준화 및 라이선스 사업화\n"
            "   ● 시청자 MOS 수집 : 60% (04/03)\n"
            "2. XTelLM: Base Private SLM 확보\n"
            "3. NDR\n"
        )
        result = extract_agenda_items(text)
        self.assertEqual(result, [
            "VQML: ITU-T 표준화 및 라이선스 사업화",
            "XTelLM: Base Private SLM 확보",
            "NDR",
        ])

    def test_circle_number_pattern(self):
        from context_loader import extract_agenda_items
        text = "① 항목 A\n② 항목 B\n"
        result = extract_agenda_items(text)
        self.assertEqual(result, ["항목 A", "항목 B"])

    def test_paren_number_pattern(self):
        from context_loader import extract_agenda_items
        text = "(1) 항목 가\n(2) 항목 나\n"
        result = extract_agenda_items(text)
        self.assertEqual(result, ["항목 가", "항목 나"])

    def test_fallback_empty_when_one_or_zero(self):
        from context_loader import extract_agenda_items
        # 1개 이하면 빈 리스트 반환
        self.assertEqual(extract_agenda_items("1. 유일한 항목"), [])
        self.assertEqual(extract_agenda_items(""), [])

    def test_sub_bullets_excluded(self):
        from context_loader import extract_agenda_items
        text = (
            "1. 메인 항목\n"
            "   - 하위 내용\n"
            "   ● 세부 항목\n"
            "2. 다른 항목\n"
        )
        result = extract_agenda_items(text)
        self.assertEqual(result, ["메인 항목", "다른 항목"])
```

- [ ] **Step 2: 테스트 실행 → 실패 확인**

```bash
cd ~/meeting_tools && python -m pytest test_context_loader.py::ExtractAgendaItemsTest -v
```

Expected: `ImportError: cannot import name 'extract_agenda_items'`

- [ ] **Step 3: `extract_agenda_items()` 구현**

`context_loader.py`의 `extract_key_terms` 함수 위에 추가:

```python
def extract_agenda_items(text: str) -> list:
    """
    회의 안건 문서에서 최상위 항목 제목 목록을 추출한다.
    번호 패턴: '1.', '①', '(1)' 등으로 시작하는 들여쓰기 없는 줄.
    파싱 결과 0~1개이면 빈 리스트 반환 (안건 구조 없는 것으로 처리).
    """
    # 번호 뒤 제목 캡처: 앞에 공백 없어야 최상위 항목
    patterns = [
        r'^\d+\.\s+(.+)',       # 1. 항목
        r'^[①②③④⑤⑥⑦⑧⑨⑩]\s*(.+)',  # ① 항목
        r'^\(\d+\)\s+(.+)',     # (1) 항목
    ]
    items = []
    for line in text.splitlines():
        # 들여쓰기 있으면 하위 항목 → 건너뜀
        if line != line.lstrip():
            continue
        for pattern in patterns:
            m = re.match(pattern, line.strip())
            if m:
                items.append(m.group(1).strip())
                break

    if len(items) <= 1:
        return []
    return items
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
cd ~/meeting_tools && python -m pytest test_context_loader.py::ExtractAgendaItemsTest -v
```

Expected: 5개 PASS

- [ ] **Step 5: commit**

```bash
cd ~/meeting_tools && git add context_loader.py test_context_loader.py
git commit -m "feat: add extract_agenda_items() to context_loader"
```

---

### Task 2: `load()` 반환값을 4-tuple로 변경

**Files:**
- Modify: `context_loader.py`
- Modify: `test_context_loader.py`

- [ ] **Step 1: 기존 반환값 테스트 → 4-tuple 기대로 변경**

`test_context_loader.py`의 `ContextLoaderTest` 내 테스트 2개를 수정:

```python
def test_load_returns_term_metadata(self):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "sample.txt"
        path.write_text(
            "AEGIS-AP WiNG VOC 기반 AI 분석 기능 개발 목표합의서",
            encoding="utf-8",
        )
        key_terms, doc_content, term_metadata, agenda_items = load([str(path)])

    self.assertTrue(key_terms)
    self.assertIn("AEGIS-AP", doc_content)
    self.assertIn("canonical_terms", term_metadata)
    self.assertIn("alias_map", term_metadata)
    self.assertIn("priority_terms", term_metadata)
    self.assertIsInstance(agenda_items, list)

def test_priority_terms_put_product_names_first(self):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "sample.txt"
        path.write_text(
            "AEGIS-AP WiNG VOC 기반 AI 분석 기능 개발",
            encoding="utf-8",
        )
        _, _, term_metadata, _ = load([str(path)])

    self.assertIn("AEGIS-AP", term_metadata["priority_terms"])
```

- [ ] **Step 2: 안건 파일 포함 테스트 추가**

`ContextLoaderTest` 클래스에 추가:

```python
def test_load_extracts_agenda_from_txt(self):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "agenda.txt"
        path.write_text(
            "1. VQML 표준화\n2. XTelLM 확보\n3. NDR 검토\n",
            encoding="utf-8",
        )
        _, _, _, agenda_items = load([str(path)])

    self.assertEqual(agenda_items, ["VQML 표준화", "XTelLM 확보", "NDR 검토"])

def test_load_returns_empty_agenda_when_no_agenda_structure(self):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "notes.txt"
        path.write_text("일반 텍스트 내용입니다.", encoding="utf-8")
        _, _, _, agenda_items = load([str(path)])

    self.assertEqual(agenda_items, [])
```

- [ ] **Step 3: 테스트 실행 → 실패 확인**

```bash
cd ~/meeting_tools && python -m pytest test_context_loader.py::ContextLoaderTest -v
```

Expected: FAIL (3-tuple unpacking 에러)

- [ ] **Step 4: `load()` 반환값 변경**

`context_loader.py`의 `load()` 함수 수정:

```python
def load(doc_paths: list) -> tuple:
    """
    문서 목록을 로드해서 (key_terms, doc_content, term_metadata, agenda_items) 반환.
    - key_terms: initial_prompt에 삽입
    - doc_content: Claude 교정용 전문
    - agenda_items: 안건 항목 목록 (없으면 [])
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
        return "", "", {"canonical_terms": [], "priority_terms": [], "alias_map": {}}, []

    doc_content = "\n\n".join(texts)
    key_terms = extract_key_terms(doc_content)
    term_metadata = extract_term_metadata(doc_content)
    agenda_items = extract_agenda_items(doc_content)
    print(f"[context_loader] 키워드 추출 완료: {key_terms[:80]}...")
    return key_terms, doc_content, term_metadata, agenda_items
```

- [ ] **Step 5: `__main__` 블록 업데이트**

`context_loader.py` 하단 `__main__` 블록 수정:

```python
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("사용법: python context_loader.py <파일1> [파일2] ...")
        sys.exit(1)
    terms, content, term_metadata, agenda_items = load(sys.argv[1:])
    print(f"\n=== 키워드 ===\n{terms}")
    print(f"\n=== 본문 (앞 500자) ===\n{content[:500]}")
    print(f"\n=== 우선 용어 ===\n{term_metadata['priority_terms']}")
    print(f"\n=== 안건 항목 ===\n{agenda_items}")
```

- [ ] **Step 6: 테스트 전체 통과 확인**

```bash
cd ~/meeting_tools && python -m pytest test_context_loader.py -v
```

Expected: 전체 PASS

- [ ] **Step 7: commit**

```bash
cd ~/meeting_tools && git add context_loader.py test_context_loader.py
git commit -m "feat: load() now returns 4-tuple with agenda_items"
```

---

## Chunk 2: pipeline.py + /meeting 스킬 Step 4

### Task 3: `pipeline.py`에 `agenda_items` 추가

**Files:**
- Modify: `pipeline.py`
- Test: `test_pipeline.py` (기존 파일 수정)

- [ ] **Step 1: 기존 pipeline 테스트 확인**

```bash
cd ~/meeting_tools && python -m pytest test_pipeline.py -v 2>&1 | head -30
```

- [ ] **Step 2: pipeline 테스트에 agenda_items 검증 추가**

`test_pipeline.py`에서 `result` JSON 검증 부분 찾아 `agenda_items` 키 확인 추가:

```python
# 기존 테스트에 아래 단언 추가 (result dict 검증하는 테스트 내부)
self.assertIn("agenda_items", result)
self.assertIsInstance(result["agenda_items"], list)
```

새 테스트도 추가:

```python
def test_run_includes_agenda_items_from_doc(self):
    """--docs에 안건 파일을 주면 result JSON에 agenda_items가 채워진다."""
    import tempfile, json
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmpdir:
        agenda_path = Path(tmpdir) / "agenda.txt"
        agenda_path.write_text(
            "1. 항목 가\n2. 항목 나\n3. 항목 다\n",
            encoding="utf-8",
        )
        # 실제 STT 없이 doc_paths만 테스트하려면 load_context를 모킹
        # 여기서는 pipeline.run()이 load_context를 통해 agenda_items를 result에 넣는지 확인
        # (integration test: 실제 오디오 없이 context_loader 경로만 검증)
        from unittest.mock import patch, MagicMock
        mock_result = ("terms", "doc content", {}, ["항목 가", "항목 나", "항목 다"])
        with patch("pipeline.load_context", return_value=mock_result):
            with patch("pipeline.run_vad", return_value=Path("/tmp/fake_vad.wav")):
                with patch("pipeline.transcribe", return_value=[]):
                    with patch("pipeline.format_transcript", return_value=""):
                        result_path = pipeline.run(
                            "/tmp/fake.wav",
                            title="테스트",
                            doc_paths=[str(agenda_path)],
                        )
        result = json.loads(Path(result_path).read_text())
        self.assertEqual(result["agenda_items"], ["항목 가", "항목 나", "항목 다"])
        Path(result_path).unlink(missing_ok=True)
```

- [ ] **Step 3: 테스트 실행 → 실패 확인**

```bash
cd ~/meeting_tools && python -m pytest test_pipeline.py -v -k "agenda" 2>&1 | tail -20
```

Expected: FAIL (`KeyError: 'agenda_items'` 또는 assertion 실패)

- [ ] **Step 4: `pipeline.py` 수정 — `load_context` 4-tuple 언패킹 + result에 추가**

`pipeline.py`의 `run()` 함수에서 두 곳 수정:

**1) `load_context` 호출 부분:**

```python
# 기존
key_terms, doc_content, term_metadata = load_context(doc_paths)
# 변경
key_terms, doc_content, term_metadata, agenda_items = load_context(doc_paths)
```

**2) `agenda_items` 초기화 (doc_paths 없을 때):**

```python
# doc_content 선언 바로 아래에 추가
doc_content = ""
term_metadata = {"canonical_terms": [], "priority_terms": [], "alias_map": {}}
agenda_items = []  # 추가
```

**3) result dict에 `agenda_items` 추가:**

```python
result = {
    "title": title,
    "date": ts,
    "speaker_count": speaker_count,
    "transcript": transcript,
    "doc_content": doc_content,
    "agenda_items": agenda_items,  # 추가
}
```

- [ ] **Step 5: 테스트 통과 확인**

```bash
cd ~/meeting_tools && python -m pytest test_pipeline.py -v
```

Expected: 전체 PASS

- [ ] **Step 6: commit**

```bash
cd ~/meeting_tools && git add pipeline.py test_pipeline.py
git commit -m "feat: pipeline result JSON includes agenda_items"
```

---

### Task 4: `/meeting` 스킬 Step 4 — 안건 인식 요약 로직

**Files:**
- Modify: `~/dotfiles/skills/meeting/SKILL.md`

이 Task는 테스트가 없다 (스킬 파일은 Claude 프롬프트 텍스트이므로 수동 검증).

- [ ] **Step 1: Step 3 JSON 구조 업데이트**

`SKILL.md`의 `### Step 3: JSON 결과 읽기` 섹션에서 JSON 구조 예시에 `agenda_items` 추가:

```json
{
  "title": "[2026-03-13] 회의",
  "date": "2026-03-13",
  "speaker_count": 4,
  "transcript": "[Speaker A]\n안녕하세요...",
  "doc_content": "참고 문서 본문 (없으면 빈 문자열)",
  "agenda_items": ["VQML 표준화", "XTelLM 확보"]
}
```

- [ ] **Step 2: Step 4 요약 섹션 교체**

`SKILL.md`의 `### Step 4: Claude 교정 + 요약` 섹션에서 액션 아이템 부분을 다음으로 교체:

```markdown
**액션 아이템 (agenda_items가 있을 때):**

JSON의 `agenda_items`가 비어있지 않으면 안건 항목별로 그룹화하여 작성한다:

```
**1. {agenda_items[0]}**
- [ ] 내용 (기한)
- [ ] 내용

**2. {agenda_items[1]}**
- [ ] 내용

**기타**
- [ ] 안건에 없지만 대화에서 명확히 도출된 액션만
```

규칙:
- 담당자 필드 없음
- 대화록에 명확히 언급된 것만 포함, 추측 금지
- 안건 항목이 있어도 대화에서 언급이 없으면 해당 항목 생략
- `기타` 섹션: 안건에 없지만 대화에서 명확히 도출된 액션만 포함 (없으면 생략)

**액션 아이템 (agenda_items가 없을 때, 기존 방식):**

```
- [ ] 내용 (기한)
- [ ] 내용
```
```

- [ ] **Step 3: 변경 사항 수동 검증**

스킬 파일을 읽어 Step 3과 Step 4가 의도한 대로 수정되었는지 확인:

```bash
grep -n "agenda_items" ~/dotfiles/skills/meeting/SKILL.md
```

Expected: Step 3 JSON 예시와 Step 4 조건 분기 두 곳에서 `agenda_items` 등장

- [ ] **Step 4: commit**

```bash
cd ~/dotfiles && git add skills/meeting/SKILL.md
git commit -m "feat: meeting skill Step 4 groups action items by agenda when available"
```

---

## 검증 시나리오

구현 완료 후 실제 사용 흐름으로 검증:

```bash
# 안건 파일 있을 때: agenda_items 채워지는지 확인
cd ~/meeting_tools && python context_loader.py ~/Downloads/2026-03-24.txt
# → "=== 안건 항목 ===" 섹션에 항목 목록 출력되면 성공

# pipeline result JSON 확인
python pipeline.py ~/meetings/some.wav "테스트 회의" --docs ~/Downloads/2026-03-24.txt
# → /tmp/meeting_*.json 파일에 "agenda_items": [...] 포함되면 성공
```
