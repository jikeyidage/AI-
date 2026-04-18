#!/bin/bash
set -e

cd "$(dirname "$0")"

MODEL=${MODEL_PATH:-"/mnt/model/Qwen3-32B"}
MODEL_NAME=$(basename "$MODEL")
VLLM_PORT=8000

echo "[run] Starting vLLM server on port ${VLLM_PORT}..."
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --served-model-name "$MODEL_NAME" \
    --tensor-parallel-size 4 \
    --port "$VLLM_PORT" \
    --enable-prefix-caching \
    --gpu-memory-utilization 0.95 \
    --max-model-len 8192 \
    --disable-log-requests \
    --disable-log-stats \
    &
VLLM_PID=$!

echo "[run] Waiting for vLLM to be ready..."
for i in $(seq 1 120); do
    if curl -s "http://localhost:${VLLM_PORT}/v1/models" > /dev/null 2>&1; then
        echo "[run] vLLM ready after ${i}s"
        break
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "[run] ERROR: vLLM process died"
        exit 1
    fi
    sleep 1
done

echo "[run] Starting client..."
python client.py &
CLIENT_PID=$!

cleanup() {
    echo "[run] Shutting down..."
    kill $CLIENT_PID 2>/dev/null || true
    kill $VLLM_PID 2>/dev/null || true
    wait
}
trap cleanup EXIT INT TERM

wait $CLIENT_PID
