# meeting-tools

회의 녹음 파일을 전처리하고, STT와 선택적 화자 분리를 거쳐 회의 대화록 JSON을 만드는 Python 도구 모음입니다.

핵심 흐름:

1. `record.py`로 마이크 입력을 녹음
2. `pipeline.py`로 VAD 전처리, Whisper STT, 선택적 pyannote 화자 분리 실행
3. 결과를 JSON으로 저장
4. 상위 스킬 레이어에서 Claude 요약과 Notion 업로드 수행

## 구성

- `record.py`: 마이크 녹음 후 WAV 저장
- `pipeline.py`: 전체 파이프라인 오케스트레이션
- `transcribe.py`: faster-whisper STT + pyannote 화자 분리
- `context_loader.py`: 참고 문서에서 키워드/본문 추출
- `summarize.py`: Anthropic API 기반 요약 도우미
- `notion_upload.py`: Notion API 업로드 예제
- `docs/`: 설계 문서와 구현 계획

## 요구 사항

- Python 3.9+
- `ffmpeg`
- Hugging Face 토큰
- Anthropic API 키

macOS 기준 `ffmpeg` 설치 예시:

```bash
brew install ffmpeg
```

## 설치

```bash
git clone https://github.com/GeonheeYe/meeting-tools.git ~/meeting_tools
cd ~/meeting_tools
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 환경 변수

공개 레포에는 실제 `.env`를 포함하지 않습니다. 샘플은 `.env.example`을 참고하세요.

```bash
cp .env.example .env
```

`dotfiles`를 함께 쓰는 환경에서는 `~/dotfiles/meeting_tools/.env`를 기준값으로 유지하고, 필요하면 아래처럼 연결해서 사용합니다.

```bash
ln -sf ~/dotfiles/meeting_tools/.env ~/meeting_tools/.env
```

필수 변수:

```env
HF_TOKEN=YOUR_HF_TOKEN_HERE
ANTHROPIC_API_KEY=YOUR_ANTHROPIC_API_KEY_HERE
```

## 사용법

녹음:

```bash
python3 record.py
```

기본 처리:

```bash
python3 pipeline.py ~/meetings/audio.wav "스프린트 회의"
```

화자 분리 + 컨텍스트 + 참고 문서:

```bash
python3 pipeline.py ~/meetings/audio.wav "XQBot 리뷰" \
  --speakers 2 \
  --context "BIRD, GRPO" \
  --docs ~/docs/agenda.txt ~/docs/notes.pdf
```

결과:

- `/tmp/meeting_YYYYMMDD_HHMMSS.json`
- 원본 오디오와 같은 디렉토리의 `*.json`
- VAD 처리된 `*_vad.wav`

## 문서

- 설계: `docs/specs/2026-03-18-context-aware-transcription-design.md`
- 계획: `docs/plans/2026-03-18-context-aware-transcription.md`

## 운영 메모

- 이 레포는 실행 코드와 문서의 소스 오브 트루스입니다.
- Codex/Claude용 오케스트레이션 지침은 `dotfiles`의 `skills/meeting/SKILL.md`에서 관리합니다.
- 민감한 `.env` 값은 공개 레포가 아니라 `dotfiles` 쪽에서 관리합니다.
