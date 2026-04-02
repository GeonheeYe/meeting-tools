# Hardware-Aware Setup 설계

**날짜**: 2026-04-03
**상태**: 승인됨

## 목표

`bash setup.sh` 한 방에 하드웨어를 감지하고 최적 패키지만 설치한다.
지원 환경: macOS (Apple Silicon / Intel), Linux, WSL. 네이티브 Windows 미지원.

## 구조

### 파일 변경

```
meeting_tools/
├── setup.sh                      # 신규 — 진입점
├── requirements-common.txt       # 신규 — 모든 환경 공통
├── requirements-mlx.txt          # 신규 — Apple Silicon 전용
├── requirements-cuda.txt         # 신규 — NVIDIA GPU
├── requirements-cpu.txt          # 신규 — CPU fallback
└── requirements.txt              # 유지 — 전체 설치용 fallback
```

### setup.sh 흐름

```
1. Python 3.9+ 확인 → 미충족 시 중단 + 안내
2. ffmpeg 감지
   - 없으면: macOS → brew install ffmpeg / Linux → sudo apt install ffmpeg
3. venv 생성 (.venv/) + pip upgrade
4. Python 한 줄로 하드웨어 백엔드 감지
   - transcribe.py의 detect_hw_config() 재사용
   - 감지 결과: "mlx" | "cuda" | "cpu"
5. 백엔드별 pip install
   - mlx  → requirements-common.txt + requirements-mlx.txt
   - cuda → requirements-common.txt + requirements-cuda.txt
   - cpu  → requirements-common.txt + requirements-cpu.txt
6. .env 설정
   - ~/dotfiles/meeting_tools/.env 있으면 symlink
   - 없고 .env.example 있으면 복사 후 편집 안내
   - 둘 다 없으면 경고만
7. HF_TOKEN / ANTHROPIC_API_KEY 값 확인
   - 비어있으면 노란색 경고 + .env 경로 안내
8. 완료 요약 출력 (백엔드, 모델, 다음 단계)
```

### requirements 파일 분리

**requirements-common.txt**
```
sounddevice==0.4.6
scipy>=1.11.0
pyannote.audio==3.1.1
anthropic>=0.20.0
python-dotenv==1.0.0
numpy>=1.24.0
pdfplumber>=0.10.0
python-docx>=1.1.0
openpyxl>=3.1.0
```

**requirements-mlx.txt**
```
mlx-whisper>=0.4.0
```

**requirements-cuda.txt**
```
faster-whisper>=1.0.0
# CUDA 드라이버는 별도 설치 필요 (cuDNN, CUDA Toolkit)
```

**requirements-cpu.txt**
```
faster-whisper>=1.0.0
```

## 하드웨어 감지 위임

setup.sh에서 Python으로 하드웨어 감지:

```bash
BACKEND=$(python3 -c "
import sys
sys.path.insert(0, '.')
from transcribe import detect_hw_config
cfg = detect_hw_config()
print(cfg['backend'])
")
```

`detect_hw_config()`는 이미 transcribe.py에 구현되어 있으므로 중복 없이 재사용한다.
단, venv 활성화 전에는 import가 안 되므로 공통 패키지 설치 후 감지 순서로 실행한다.

## 오류 처리

- Python < 3.9 → 즉시 종료, 버전 업그레이드 안내
- brew/apt 없음 → 수동 설치 링크 출력 후 종료
- venv 생성 실패 → 에러 메시지 + 종료
- 하드웨어 감지 실패 → cpu fallback으로 진행
- .env 없음 → 경고만, 설치는 계속

## 검증 기준

1. macOS Apple Silicon에서 `bash setup.sh` 실행 시 mlx-whisper만 설치됨
2. Linux/WSL CPU에서 실행 시 faster-whisper만 설치됨 (mlx-whisper 미설치)
3. NVIDIA CUDA 환경에서 faster-whisper[cuda]가 설치됨
4. dotfiles .env가 있으면 symlink, 없으면 .env.example 복사
5. HF_TOKEN 미설정 시 경고 출력
6. Python 3.8에서 실행 시 명확한 에러 메시지와 함께 종료
