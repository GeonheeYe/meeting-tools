# 안건 항목별 액션 아이템 구조화 설계

## 목적

txt 형식의 회의 안건 파일이 제공될 때, 액션 아이템을 안건 항목별로 그룹화해서 출력한다.
안건 파일이 없으면 기존 flat 형식을 유지한다.

## 변경 범위

총 3개 파일 수정. 파이프라인 흐름은 그대로 유지.

---

## 1. `context_loader.py` — 안건 구조 추출

### 추가 함수: `extract_agenda_items(text: str) -> list[str]`

회의 안건 문서에서 항목 제목 목록을 추출한다.

**파싱 규칙:**
- 번호 패턴: `1.`, `2.`, `①`, `(1)` 등으로 시작하는 줄
- 들여쓰기 없는 최상위 항목만 추출 (하위 bullet `●`, `-` 등은 제외)
- 항목 제목에서 번호 제거 후 반환

**예시 입력:**
```
1. VQML: ITU-T 표준화 및 라이선스 사업화
   ● 시청자 MOS 수집 : 60% (04/03)
2. XTelLM: Base Private SLM 확보
3. NDR
```

**예시 출력:**
```python
["VQML: ITU-T 표준화 및 라이선스 사업화", "XTelLM: Base Private SLM 확보", "NDR"]
```

**폴백:** 파싱 결과가 0개이거나 1개이면 빈 리스트 반환 → 안건 구조 없는 것으로 처리.

### `load()` 반환값 변경

기존: `(key_terms, doc_content, term_metadata)`
변경: `(key_terms, doc_content, term_metadata, agenda_items)`

---

## 2. `pipeline.py` — result JSON에 `agenda_items` 추가

`load_context()` 호출 결과에서 `agenda_items`를 받아 result dict에 포함한다.

```python
result = {
    "title": title,
    "date": ts,
    "speaker_count": speaker_count,
    "transcript": transcript,
    "doc_content": doc_content,
    "agenda_items": agenda_items,  # 추가. 없으면 []
}
```

---

## 3. `/meeting` 스킬 Step 4 — 안건 인식 요약

JSON의 `agenda_items`가 비어있지 않으면, Claude에게 다음 구조로 액션 아이템 작성을 지시한다.

### 액션 아이템 출력 형식 (agenda_items 있을 때)

```
**1. VQML: ITU-T 표준화**
- [ ] 내용 (기한)
- [ ] 내용

**2. XTelLM**
- [ ] 내용 (기한)

**기타**
- [ ] 안건에 명시되지 않은 항목들
```

### 액션 아이템 출력 형식 (agenda_items 없을 때, 기존 방식)

```
- [ ] 내용 (기한)
- [ ] 내용
```

### 규칙
- 담당자 필드 없음
- 대화록에 명확히 언급된 것만 포함, 추측 금지
- 안건 항목이 있어도 대화에서 언급이 없으면 해당 항목 생략
- `기타` 섹션: 안건에 없지만 대화에서 명확히 도출된 액션만

---

## 데이터 흐름

```
txt 안건 파일
    → context_loader.load()
        → extract_agenda_items()  [신규]
        → agenda_items: list[str]
    → pipeline result JSON
        → agenda_items 포함
    → /meeting 스킬 Step 4
        → agenda_items 있으면 그룹화 요약
        → 없으면 기존 flat 요약
```

---

## 비변경 범위

- `transcribe.py`, `record.py`, `summarize.py` 무변경
- 파이프라인 VAD/STT/화자분리 로직 무변경
- Notion 업로드 로직 무변경
- 안건 파일 없는 기존 사용법 완전 호환
