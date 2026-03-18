# Context-Aware Transcription 설계

**날짜**: 2026-03-18
**상태**: 승인됨

## 목표

사람이 배경 지식이 있으면 뭉개진 발음도 알아듣듯이, 회의 참고 문서를 미리 주입해서 STT 정확도를 높인다. 두 단계로 동작한다:

1. **STT 전**: 참고 문서에서 추출한 키워드를 `initial_prompt`에 주입 → Whisper 1차 인식률 향상
2. **STT 후**: Claude가 전체 대화록 + 참고 문서를 보고 오류 교정 + 요약을 한 번에 처리

## 파이프라인 흐름

```
/meeting audio.wav 제목 [N명] [키워드] [참고문서들]

1. context_loader  → 문서에서 키워드 추출 → initial_prompt 구성
2. VAD             → 무음 제거 (기존)
3. Whisper         → STT (풍부한 initial_prompt로 1차 정확도 향상)
4. pyannote        → [--speakers N 있을 때만] 화자 분리
5. Claude (스킬)   → 대화록 + 참고 문서 보고 교정 + 요약 한 번에
6. Notion 업로드   → (기존)
```

## 컴포넌트 상세

### 1. `context_loader.py` (신규)

- **입력**: 파일 경로 목록 (PDF, TXT, DOCX, 이미지 등)
- **출력**:
  - `key_terms: str` — 쉼표 구분 핵심 용어, `initial_prompt`에 삽입
  - `doc_content: str` — Claude 교정 단계에 전달할 문서 전문(또는 요약)
- **방법**: 텍스트 추출 → 명사/고유명사 위주 키워드 추출 (LLM 불필요, 규칙 기반)
- **지원 포맷**: `.txt`, `.md`, `.pdf`(pdfplumber), `.docx`(python-docx)

### 2. `pipeline.py` (수정)

```
변경 전: run(audio_path, title, num_speakers, context)
변경 후: run(audio_path, title, num_speakers, context, doc_paths)
```

- `doc_paths`가 있으면 `context_loader`로 키워드 추출 → 기존 `context`에 append
- `num_speakers`가 없으면 pyannote 생략 (화자 분리 선택적)
- 결과 JSON에 `doc_content` 필드 추가 (Claude 교정용)

```json
{
  "title": "...",
  "date": "...",
  "speaker_count": 2,
  "transcript": "...",
  "doc_content": "참고 문서 전문 또는 요약"
}
```

### 3. `SKILL.md` (수정)

**파싱 변경**: 문서 파일 경로 인식 추가

```
/meeting audio.wav 제목 2명 키워드 doc1.pdf doc2.txt
```

**Step 4 변경 (Claude 교정 + 요약 통합)**:

기존: transcript만 보고 요약
변경: transcript + `doc_content` 보고 아래 순서로 처리
1. STT 오류 교정 (도메인 용어, 고유명사 위주)
2. 교정된 대화록 기반으로 요약 / 액션 아이템 / 주요 결정사항 작성

## 화자 분리 정책

- `--speakers N` 옵션 있을 때만 pyannote 실행 (기존 동작 유지)
- 옵션 없으면 화자 분리 생략 → 처리 시간 대폭 단축
- Claude 교정 단계에서 대화 패턴으로 화자 전환 추론 가능

## 변경 파일 요약

| 파일 | 변경 유형 | 내용 |
|------|----------|------|
| `context_loader.py` | 신규 | 문서 → 키워드 + 본문 추출 |
| `pipeline.py` | 수정 | doc_paths 파라미터, diarization 조건부, doc_content JSON 저장 |
| `SKILL.md` | 수정 | 문서 파일 파싱, Claude 교정+요약 통합 |

## 사용 예시

```bash
# 기본 (문서 없음, 화자 분리 없음)
/meeting audio.wav 스프린트 회의

# 문서 첨부 + 화자 분리
/meeting audio.wav XQBot 논문 리뷰 2명 schema_linking.pdf previous_meeting.txt

# 키워드 + 문서 혼합
/meeting audio.wav 목표합의서 3명 RAG,AEGIS agenda.txt
```

## 제외 항목 (YAGNI)

- 실시간 미리보기 — 목표가 정확도이지 실시간 텍스트가 아님
- anthropic SDK 직접 호출 — Claude는 스킬 레이어에서 처리
- 청크 단위 스트리밍 — 배치 처리로 충분
