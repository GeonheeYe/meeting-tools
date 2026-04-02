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

# ─── Step 1: Python 3.9+ 확인 ───────────────────────────────────────────────

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

# ─── Step 2: ffmpeg 확인 및 설치 ────────────────────────────────────────────

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

# ─── Step 3: venv 생성 ──────────────────────────────────────────────────────

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

# ─── Step 4: 하드웨어 감지 (순수 bash) ─────────────────────────────────────

info "하드웨어 감지 중..."
_OS=$(uname -s)
_ARCH=$(uname -m)

if [[ "$_OS" == "Darwin" && "$_ARCH" == "arm64" ]]; then
    BACKEND="mlx"
    # RAM 크기 감지 (macOS)
    RAM_BYTES=$(sysctl -n hw.memsize 2>/dev/null || echo 0)
    RAM_GB=$(( RAM_BYTES / 1024 / 1024 / 1024 ))
    HW_LABEL="Apple Silicon (Metal, ${RAM_GB}GB RAM)"
elif command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null 2>&1; then
    BACKEND="cuda"
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "NVIDIA GPU")
    VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 || echo 0)
    VRAM_GB=$(( VRAM_MB / 1024 ))
    HW_LABEL="NVIDIA ${GPU_NAME} (${VRAM_GB}GB VRAM)"
else
    BACKEND="cpu"
    _OS2=$(uname -s)
    if [[ "$_OS2" == "Darwin" ]]; then
        RAM_BYTES=$(sysctl -n hw.memsize 2>/dev/null || echo 0)
        RAM_GB=$(( RAM_BYTES / 1024 / 1024 / 1024 ))
    else
        RAM_KB=$(grep MemTotal /proc/meminfo 2>/dev/null | awk '{print $2}' || echo 0)
        RAM_GB=$(( RAM_KB / 1024 / 1024 ))
    fi
    HW_LABEL="CPU (${RAM_GB}GB RAM)"
fi

info "감지된 환경: ${HW_LABEL} → backend=${BACKEND}"

# ─── Step 5: 백엔드별 pip install ───────────────────────────────────────────

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

# ─── Step 6: .env 설정 ──────────────────────────────────────────────────────

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
    warn "  cp .env.example .env"
fi

# ─── Step 7: 환경 변수 확인 ─────────────────────────────────────────────────

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
    warn "ANTHROPIC_API_KEY가 설정되지 않았습니다. Claude Code 자체를 사용 중이면 불필요합니다."
    MISSING=1
else
    success "ANTHROPIC_API_KEY 확인"
fi

[[ "$MISSING" -eq 1 ]] && warn ".env 경로: $ENV_FILE"

# ─── Step 8: 완료 요약 출력 ─────────────────────────────────────────────────

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
echo -e "  .venv/bin/python3 pipeline.py <오디오파일> [제목] [--speakers N]"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
