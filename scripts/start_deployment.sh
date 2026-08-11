#!/bin/sh
set -eu

# Cloud hosts expose one public port. FastAPI stays private and Streamlit is public.
api_port="${INTERNAL_API_PORT:-8000}"
public_port="${PORT:-7860}"
api_url="http://127.0.0.1:${api_port}"

api_pid=""
frontend_pid=""

shutdown() {
    trap - EXIT INT TERM
    if [ -n "${frontend_pid}" ]; then
        kill "${frontend_pid}" 2>/dev/null || true
    fi
    if [ -n "${api_pid}" ]; then
        kill "${api_pid}" 2>/dev/null || true
    fi
    wait 2>/dev/null || true
}

trap shutdown EXIT INT TERM

python -m uvicorn api.main:app \
    --host 127.0.0.1 \
    --port "${api_port}" &
api_pid=$!

# TensorFlow model loading can take a while on a small cloud CPU.
attempt=1
while [ "${attempt}" -le 120 ]; do
    if python -c "import urllib.request; urllib.request.urlopen('${api_url}/health', timeout=2).read()" 2>/dev/null; then
        break
    fi
    if ! kill -0 "${api_pid}" 2>/dev/null; then
        echo "FastAPI stopped before it became healthy." >&2
        exit 1
    fi
    attempt=$((attempt + 1))
    sleep 1
done

if [ "${attempt}" -gt 120 ]; then
    echo "FastAPI did not become healthy within 120 seconds." >&2
    exit 1
fi

API_URL="${api_url}" python -m streamlit run frontend/streamlit_app.py \
    --server.address=0.0.0.0 \
    --server.port="${public_port}" \
    --browser.gatherUsageStats=false &
frontend_pid=$!

# End the container if either service unexpectedly stops.
while kill -0 "${api_pid}" 2>/dev/null && kill -0 "${frontend_pid}" 2>/dev/null; do
    sleep 2
done

echo "A deployment process stopped; shutting down the container." >&2
exit 1
