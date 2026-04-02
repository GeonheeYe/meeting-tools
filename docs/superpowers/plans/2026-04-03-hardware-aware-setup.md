# Hardware-Aware Setup 구현 플랜

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `bash setup.sh` 한 방에 하드웨어를 감지하고 최적 패키지만 설치한다.

**Architecture:** setup.sh가 순수 bash로 OS/arch/GPU를 감지하고, 백엔드별 requirements 파일을 골라 pip install한다. Python import 없이 `uname` + `nvidia-smi`로 감지하므로 venv 생성 전에도 동작한다.

**Tech Stack:** bash, pip, brew (macOS), apt (Linux/WSL)

---

## Chunk 1: requirements 파일 분리

### Task 1: requirements 파일 4개 생성

**Files:**
- Create: `meeting_tools/requirements-common.txt`
- Create: `meeting_tools/requirements-mlx.txt`
- Create: `meeting_tools/requirements-cuda.txt`
- Create: `meeting_tools/requirements-cpu.txt`
- Keep: `meeting_tools/requirements.txt` (전체 설치용 fallback, 수정 없음)

- [ ] **Step 1: requirements-common.txt 생성**

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

- [ ] **Step 2: requirements-mlx.txt 생성**

```
mlx-whisper>=0.4.0
```

- [ ] **Step 3: requirements-cuda.txt 생성**

```
faster-whisper>=1.0.0
# CUDA Toolkit / cuDNN은 시스템에 별도 설치 필요
# https://docs.nvidia.com/cuda/cuda-installation-guide-linux/
```

- [ ] **Step 4: requirements-cpu.txt 생성**

```
faster-whisper>=1.0.0
```

- [ ] **Step 5: 기존 requirements.txt가 그대로인지 확인**

```bash
cat ~/meeting_tools/requirements.txt
```
Expected: 기존 전체 목록 그대로 (mlx-whisper + faster-whisper 모두 포함).
수정 없음 — 수동 또는 CI 전체 설치용 fallback으로 유지.

- [ ] **Step 6: commit**

```bash
cd ~/meeting_tools
git add requirements-common.txt requirements-mlx.txt requirements-cuda.txt requirements-cpu.txt
git commit -m "feat: split requirements by hardware backend"
```

---

## Chunk 2: setup.sh 구현

### Task 2: setup.sh 골격 + Python 버전 체크

**Files:**
- Create: `meeting_tools/setup.sh`

- [ ] **Step 1: setup.sh 파일 생성 (골격 + 색상 헬퍼)**

```bash
#!/usr/bin/env bash
set -euo pipefail

# 색상 출력 헬퍼
RED='\033[0;31m'; YELLOW='\033[1;33m'; GREEN='\033[0;32m'; BOLD='\033[1m'; NC='\033[0m'
info()    { echo -e "${BOLD}[setup]${NC} $*"; }
success() { echo -e "${GREEN}[setup]${NC} $*"; }
warn()    { echo -e "${YELLOW}[warn]${NC}  $*"; }
error()   { echo -e "${RED}[error]${NC} $*"; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
```

- [ ] **Step 2: Python 3.9+ 확인 로직 추가**

```bash
# Python 3.9+ 확인
info "Python 버전 확인 중..."
PYTHON=$(command -v python3 || true)
[[ -z "$PYTHON" ]] && error "python3를 찾을 수 없습니다. https://www.python.org/downloads/ 에서 3.9+ 설치 후 재시도하세요."

PY_VER=$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$("$PYTHON" -c "import sys; print(sys.version_info.major)")
PY_MINOR=$("$PYTHON" -c "import sys; print(sys.version_info.minor)")

if [[ "$PY_MAJOR" -lt 3 ]] || { [[ "$PY_MAJOR" -eq 3 ]] && [[ "$PY_MINOR" -lt 9 ]]; }; then
    error "Python 3.9 이상이 필요합니다. 현재: ${PY_VER}\nhttps://www.python.org/downloads/"
fi
success "Python ${PY_VER} 확인"
```

- [ ] **Step 3: 수동 검증 — Python 3.8 흉내**

```bash
bash -c '
  PY_MINOR=8
  if [[ "3" -lt 3 ]] || { [[ "3" -eq 3 ]] && [[ "$PY_MINOR" -lt 9 ]]; }; then
    echo "FAIL: 3.8 감지됨 — 정상"
  fi
'
```
Expected: `FAIL: 3.8 감지됨 — 정상` 출력.

### Task 3: ffmpeg 자동 설치

- [ ] **Step 4: ffmpeg 감지 + 자동 설치 로직 추가**

```bash
# ffmpeg 확인 및 설치
info "ffmpeg 확인 중..."
if ! command -v ffmpeg &>/dev/null; then
    warn "ffmpeg가 없습니다. 설치를 시도합니다..."
    OS_TYPE=$(uname -s)
    if [[ "$OS_TYPE" == "Darwin" ]]; then
        command -v brew &>/dev/null || error "Homebrew가 없습니다. https://brew.sh 설치 후 재시도하세요."
        brew install ffmpeg
    elif [[ "$OS_TYPE" == "Linux" ]]; then
        command -v apt-get &>/dev/null || error "apt-get이 없습니다. 수동으로 ffmpeg를 설치하세요: https://ffmpeg.org/download.html"
        sudo apt-get update -qq && sudo apt-get install -y ffmpeg
    else
        error "지원하지 않는 OS입니다: ${OS_TYPE}. 수동으로 ffmpeg를 설치하세요."
    fi
fi
success "ffmpeg $(ffmpeg -version 2>&1 | head -1 | awk '{print $3}') 확인"
```

### Task 4: venv 생성

- [ ] **Step 5: venv 생성 + pip upgrade 로직 추가**

```bash
# venv 생성
VENV_DIR="$SCRIPT_DIR/.venv"
if [[ ! -d "$VENV_DIR" ]]; then
    info ".venv 생성 중..."
    "$PYTHON" -m venv "$VENV_DIR" || error "venv 생성에 실패했습니다."
    success ".venv 생성 완료"
else
    info ".venv 이미 존재합니다."
fi

VENV_PY="$VENV_DIR/bin/python3"
VENV_PIP="$VENV_DIR/bin/pip"

"$VENV_PIP" install --upgrade pip --quiet
```

### Task 5: 하드웨어 감지 (순수 bash)

- [ ] **Step 6: 하드웨어 감지 로직 추가**

`detect_hw_config()`를 import하지 않고 순수 bash로 감지한다.
venv 설치 전에도 동작해야 하므로 `uname` + `nvidia-smi`만 사용한다.

```bash
# 하드웨어 감지
info "하드웨어 감지 중..."
_OS=$(uname -s)
_ARCH=$(uname -m)

if [[ "$_OS" == "Darwin" && "$_ARCH" == "arm64" ]]; then
    BACKEND="mlx"
    HW_LABEL="Apple Silicon (Metal)"
elif command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null 2>&1; then
    BACKEND="cuda"
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "NVIDIA GPU")
    HW_LABEL="NVIDIA GPU: ${GPU_NAME}"
else
    BACKEND="cpu"
    HW_LABEL="CPU fallback"
fi

info "감지된 환경: ${HW_LABEL} → backend=${BACKEND}"
```

- [ ] **Step 7: 수동 검증 — arm64 감지**

```bash
bash -c '
  _OS=Darwin; _ARCH=arm64
  if [[ "$_OS" == "Darwin" && "$_ARCH" == "arm64" ]]; then echo "mlx"; fi
'
```
Expected: `mlx`

```bash
bash -c '
  _OS=Linux; _ARCH=x86_64
  if [[ "$_OS" == "Darwin" && "$_ARCH" == "arm64" ]]; then echo "mlx"
  else echo "cpu"; fi
'
```
Expected: `cpu`

### Task 6: 백엔드별 pip install

- [ ] **Step 8: 백엔드별 설치 로직 추가**

```bash
# 백엔드별 pip install
info "패키지 설치 중 (backend=${BACKEND})..."

"$VENV_PIP" install -r "$SCRIPT_DIR/requirements-common.txt" --quiet

case "$BACKEND" in
    mlx)
        "$VENV_PIP" install -r "$SCRIPT_DIR/requirements-mlx.txt" --quiet
        ;;
    cuda)
        warn "CUDA Toolkit / cuDNN이 시스템에 설치되어 있어야 합니다."
        warn "설치 안내: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/"
        "$VENV_PIP" install -r "$SCRIPT_DIR/requirements-cuda.txt" --quiet
        ;;
    cpu)
        "$VENV_PIP" install -r "$SCRIPT_DIR/requirements-cpu.txt" --quiet
        ;;
esac

success "패키지 설치 완료"
```

---

## Chunk 3: .env 설정 + 완료 출력

### Task 7: .env 설정

**Files:**
- Modify: `meeting_tools/setup.sh` (append)

- [ ] **Step 1: .env 설정 로직 추가**

```bash
# .env 설정
ENV_FILE="$SCRIPT_DIR/.env"
DOTFILES_ENV="$HOME/dotfiles/meeting_tools/.env"
ENV_EXAMPLE="$SCRIPT_DIR/.env.example"

info ".env 설정 확인 중..."

if [[ -f "$ENV_FILE" ]] || [[ -L "$ENV_FILE" ]]; then
    info ".env 이미 존재합니다."
elif [[ -f "$DOTFILES_ENV" ]]; then
    ln -sf "$DOTFILES_ENV" "$ENV_FILE"
    success ".env → dotfiles symlink 생성: $DOTFILES_ENV"
elif [[ -f "$ENV_EXAMPLE" ]]; then
    cp "$ENV_EXAMPLE" "$ENV_FILE"
    warn ".env.example을 복사했습니다. 아래 항목을 직접 편집하세요:"
    warn "  $ENV_FILE"
else
    warn ".env 파일이 없습니다. 수동으로 생성하세요:"
    warn "  cp .env.example .env  # .env.example 참고"
fi
```

- [ ] **Step 2: HF_TOKEN / ANTHROPIC_API_KEY 확인 로직 추가**

```bash
# 환경 변수 확인
if [[ -f "$ENV_FILE" ]]; then
    set -a; source "$ENV_FILE"; set +a
fi

echo ""
info "환경 변수 확인..."
MISSING=0

if [[ -z "${HF_TOKEN:-}" ]]; then
    warn "HF_TOKEN이 설정되지 않았습니다. 화자 분리(pyannote)를 사용하려면 필요합니다."
    warn "  https://huggingface.co/settings/tokens 에서 발급 후 .env에 추가하세요."
    MISSING=1
else
    success "HF_TOKEN 확인"
fi

if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
    warn "ANTHROPIC_API_KEY가 설정되지 않았습니다. Claude 요약 기능을 사용하려면 필요합니다."
    MISSING=1
else
    success "ANTHROPIC_API_KEY 확인"
fi

[[ "$MISSING" -eq 1 ]] && warn ".env 경로: $ENV_FILE"
```

### Task 8: 완료 요약 출력

- [ ] **Step 3: 완료 요약 출력 추가**

```bash
# 완료 요약
echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}  설치 완료${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "  환경:   ${HW_LABEL}"
echo -e "  백엔드: ${BACKEND}"
echo -e "  venv:   ${VENV_DIR}"
echo ""
echo -e "다음 단계:"
echo -e "  source .venv/bin/activate"
echo -e "  python3 pipeline.py <오디오파일> [제목] [--speakers N]"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
```

- [ ] **Step 4: 실행 권한 부여 + 현재 머신에서 end-to-end 테스트**

```bash
chmod +x ~/meeting_tools/setup.sh
cd ~/meeting_tools && bash setup.sh
```

Expected 출력:
```
[setup] Python 3.x.x 확인
[setup] ffmpeg x.x 확인
[setup] 하드웨어 감지 중...
[setup] 감지된 환경: Apple Silicon (Metal) → backend=mlx
[setup] 패키지 설치 중 (backend=mlx)...
[setup] 패키지 설치 완료
...
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  설치 완료
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  환경:   Apple Silicon (Metal)
  백엔드: mlx
```

- [ ] **Step 5: mlx-whisper만 설치됐는지 확인 (faster-whisper 없어야 함)**

```bash
.venv/bin/pip list | grep -E "mlx|faster"
```

Expected:
```
mlx-whisper    x.x.x   ← 있어야 함
```
`faster-whisper`가 목록에 없으면 정상.

- [ ] **Step 6: commit**

```bash
cd ~/meeting_tools
git add setup.sh
git commit -m "feat: add hardware-aware setup.sh (mlx/cuda/cpu auto-detect)"
```

---

## 완료 체크리스트

- [ ] `bash setup.sh` → Apple Silicon에서 mlx-whisper만 설치됨
- [ ] Python < 3.9에서 명확한 에러 메시지로 종료됨
- [ ] ffmpeg 없을 때 자동 설치 시도
- [ ] dotfiles .env 있으면 symlink, 없으면 .env.example 복사
- [ ] HF_TOKEN / ANTHROPIC_API_KEY 미설정 시 노란색 경고 출력
- [ ] 완료 요약에 백엔드·환경·다음 단계 출력
