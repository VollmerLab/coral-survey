#!/usr/bin/env bash
# Run this script on Athene after cloning the repo.
# Usage: bash deploy.sh [PORT]
set -e

PORT=${1:-8080}
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$REPO_DIR/logs"
PID_FILE="$REPO_DIR/logs/server.pid"

mkdir -p "$LOG_DIR"
mkdir -p "$REPO_DIR/models"
mkdir -p "$REPO_DIR/data"

echo "=== Coral Survey — Athene deployment ==="
echo "Repo:  $REPO_DIR"
echo "Port:  $PORT"
echo ""

# ── Python venv ───────────────────────────────────────────────────
if [ ! -d "$REPO_DIR/venv" ]; then
    echo "[1/5] Creating Python venv..."
    python3 -m venv "$REPO_DIR/venv"
fi
source "$REPO_DIR/venv/bin/activate"

# ── Dependencies ──────────────────────────────────────────────────
echo "[2/5] Installing requirements..."
pip install --quiet --upgrade pip
pip install --quiet -r "$REPO_DIR/requirements.txt"

# segment-anything
if ! python -c "import segment_anything" 2>/dev/null; then
    echo "      Installing segment-anything..."
    pip install --quiet git+https://github.com/facebookresearch/segment-anything.git
fi

# PyTorch — detect CUDA
if ! python -c "import torch" 2>/dev/null; then
    echo "      Installing PyTorch..."
    CUDA_VER=$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+\.[0-9]+' | head -1 | tr -d '.')
    if [ -n "$CUDA_VER" ]; then
        echo "      CUDA $CUDA_VER detected"
        pip install --quiet torch torchvision --index-url "https://download.pytorch.org/whl/cu${CUDA_VER}"
    else
        echo "      No CUDA found — installing CPU-only torch"
        pip install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cpu
    fi
fi

# ── SAM model weights ─────────────────────────────────────────────
SAM_PATH="$REPO_DIR/models/sam_vit_b.pth"
if [ ! -f "$SAM_PATH" ]; then
    echo "[3/5] Downloading SAM model weights (~375 MB)..."
    wget -q --show-progress -O "$SAM_PATH" \
        https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
else
    echo "[3/5] SAM weights already present — skipping"
fi

# ── Stop any existing server ──────────────────────────────────────
echo "[4/5] Stopping any existing server..."
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    kill "$OLD_PID" 2>/dev/null && echo "      Stopped PID $OLD_PID" || true
    rm -f "$PID_FILE"
fi
# Also kill by port just in case
fuser -k "${PORT}/tcp" 2>/dev/null || true

# ── Start server ──────────────────────────────────────────────────
echo "[5/5] Starting server on port $PORT..."
cd "$REPO_DIR"
nohup python -m uvicorn web.main:app --host 0.0.0.0 --port "$PORT" \
    > "$LOG_DIR/server.log" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" > "$PID_FILE"

sleep 2
if kill -0 "$SERVER_PID" 2>/dev/null; then
    echo ""
    echo "✓ Server running (PID $SERVER_PID)"
    echo "  Local URL:  http://localhost:$PORT"
    echo "  SSH tunnel: ssh -L $PORT:localhost:$PORT svollmer@athene-login.hpc.fau.edu"
    echo "  Log:        tail -f $LOG_DIR/server.log"
    echo "  Stop:       kill \$(cat $PID_FILE)"
else
    echo "✗ Server failed to start. Check logs:"
    tail -20 "$LOG_DIR/server.log"
    exit 1
fi
