#!/bin/bash
set -e

cd "$(dirname "$0")"

MODEL=${MODEL_PATH:-"/mnt/model/Qwen3-32B"}
MODEL_NAME=$(basename "$MODEL")
VLLM_PORT=8000
export VLLM_PORT

# --- Quantization (Tier 3) ---------------------------------------------------
# QUANT_MODE:
#   fp8   (default) — on-the-fly FP8 for the BF16 weights + FP8 KV cache.
#                     5090 has native FP8 tensor cores. ~1.5-2x vs BF16, near-lossless.
#                     Adds ~15-60s to startup (online conversion of 32B model).
#   awq   — load a pre-quantized AWQ INT4 checkpoint. Requires AWQ_MODEL_PATH
#           to point at the weights (bundled or downloaded separately). ~2-3x
#           speedup, <1% quality loss typically.
#   none  — disable quantization (BF16 baseline).
# KV_CACHE_DTYPE overrides the KV cache format (auto|fp8). Default follows QUANT_MODE.
QUANT_MODE=${QUANT_MODE:-fp8}
KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-}
QUANT_ARGS=""
case "$QUANT_MODE" in
    fp8)
        echo "[run] Quantization: FP8 online (weights + KV cache)"
        QUANT_ARGS="--quantization fp8"
        [ -z "$KV_CACHE_DTYPE" ] && KV_CACHE_DTYPE="fp8"
        ;;
    awq)
        AWQ_MODEL_PATH=${AWQ_MODEL_PATH:-}
        if [ -z "$AWQ_MODEL_PATH" ] || [ ! -d "$AWQ_MODEL_PATH" ]; then
            echo "[run] ERROR: QUANT_MODE=awq but AWQ_MODEL_PATH is not set or missing"
            exit 1
        fi
        MODEL="$AWQ_MODEL_PATH"
        MODEL_NAME=$(basename "$MODEL")
        echo "[run] Quantization: AWQ (pre-quantized model at $MODEL)"
        QUANT_ARGS="--quantization awq_marlin"
        ;;
    none)
        echo "[run] Quantization: disabled (BF16 baseline)"
        ;;
    *)
        echo "[run] ERROR: unknown QUANT_MODE=$QUANT_MODE (expected fp8|awq|none)"
        exit 1
        ;;
esac
KV_ARGS=""
if [ -n "$KV_CACHE_DTYPE" ] && [ "$KV_CACHE_DTYPE" != "auto" ]; then
    KV_ARGS="--kv-cache-dtype $KV_CACHE_DTYPE"
    echo "[run] KV cache dtype: $KV_CACHE_DTYPE"
fi

# --- Speculative decoding draft model ----------------------------------------
# Priority for finding a draft model:
#   1. $SPEC_MODEL env var (explicit override)
#   2. ./draft_model/ bundled in the submission tarball (recommended — platform
#      is not guaranteed to have a small Qwen3 on disk; Q33 allows BYO).
#   3. If neither → disable speculative decoding.
NUM_SPEC_TOKENS=${NUM_SPEC_TOKENS:-5}
SPEC_ARGS=""
if [ -z "$SPEC_MODEL" ] && [ -d "./draft_model" ]; then
    SPEC_MODEL="$(pwd)/draft_model"
fi
if [ -n "$SPEC_MODEL" ] && [ -d "$SPEC_MODEL" ]; then
    echo "[run] Enabling speculative decoding with draft model: $SPEC_MODEL (num_speculative_tokens=$NUM_SPEC_TOKENS)"
    SPEC_ARGS="--speculative-model $SPEC_MODEL --num-speculative-tokens $NUM_SPEC_TOKENS"
else
    echo "[run] No draft model found → speculative decoding disabled (place one at ./draft_model/ or set SPEC_MODEL)"
fi

# --- vLLM server -------------------------------------------------------------
# Performance tuning rationale:
#   Tier 1 (BF16 baseline):
#     --max-num-seqs 16           cap concurrency to keep single-task TTFT low
#     --max-num-batched-tokens 8192  full-prompt prefill fits in one step
#     --enable-chunked-prefill    interleave prefill/decode → decode SLA protected
#     --enable-prefix-caching     identical prompt prefixes cached across tasks
#     --gpu-memory-utilization 0.95  leave headroom for activations/tp comms
#     (no --enforce-eager)        keep CUDA graphs on (~10-20% faster)
#   Tier 2: speculative decoding → $SPEC_ARGS (above)
#   Tier 3: quantization → $QUANT_ARGS, $KV_ARGS (above)
echo "[run] Starting vLLM server on port ${VLLM_PORT}..."
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --served-model-name "$MODEL_NAME" \
    --tensor-parallel-size 4 \
    --port "$VLLM_PORT" \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --max-num-seqs 16 \
    --max-num-batched-tokens 8192 \
    --gpu-memory-utilization 0.95 \
    --max-model-len 8192 \
    --disable-log-requests \
    --disable-log-stats \
    $QUANT_ARGS \
    $KV_ARGS \
    $SPEC_ARGS \
    &
VLLM_PID=$!

echo "[run] Waiting for vLLM to be ready..."
for i in $(seq 1 300); do
    if curl -s "http://localhost:${VLLM_PORT}/v1/models" > /dev/null 2>&1; then
        echo "[run] vLLM ready after ${i}s"
        break
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "[run] ERROR: vLLM process died during startup"
        exit 1
    fi
    sleep 1
done

# --- Warmup: trigger CUDA graph capture + prefix cache pre-fill --------------
# First few real requests are slow because vLLM captures CUDA graphs at various
# batch sizes on demand. Sending a handful of dummy requests now means the
# first real task hits warm graphs instead of eating that cost on the SLA clock.
echo "[run] Warming up vLLM (dummy requests)..."
python warmup.py --model "$MODEL_NAME" --port "$VLLM_PORT" || \
    echo "[run] WARN: warmup failed, continuing anyway"

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
