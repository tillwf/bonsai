#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

DEMO_VENV="demo/.venv"
OWN_VENV=".venv"

# Prefer demo's venv — it already has torch/diffusers/gemlite/hqq
if [ -f "$DEMO_VENV/bin/uvicorn" ]; then
    PYTHON="$DEMO_VENV/bin/python"
    UVICORN="$DEMO_VENV/bin/uvicorn"
    echo "Using demo venv: $DEMO_VENV"
else
    # Fall back to own venv (only has fastapi/uvicorn/httpx — no ML)
    if [ ! -d "$OWN_VENV" ]; then
        python3 -m venv "$OWN_VENV"
        "$OWN_VENV/bin/pip" install -q -r requirements.txt
    fi
    PYTHON="$OWN_VENV/bin/python"
    UVICORN="$OWN_VENV/bin/uvicorn"
    echo "Using own venv: $OWN_VENV (no ML libs — run 'cd demo && bash setup.sh' first)"
fi

PORT=${PORT:-3001}

cuda_status=$("$PYTHON" -c "import torch; print('GPU' if torch.cuda.is_available() else 'CPU')" 2>/dev/null || echo "unknown")
echo "Mode: $cuda_status"
echo "UI  → http://localhost:$PORT"
echo ""

"$UVICORN" app:app \
    --host 0.0.0.0 \
    --port "$PORT" \
    --timeout-keep-alive 300 \
    --log-level info
