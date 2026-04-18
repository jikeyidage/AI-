#!/bin/bash
# IMPORTANT: no `set -e`. We need to handle vLLM startup failures ourselves
# and fall back to more conservative configs rather than abort the whole run.

cd "$(dirname "$0")"

# Team credentials (AAA / 杨东升). Baked into the submission so client.py
# /register with correct identity on the real platform. Env var overrides
# these if set externally (e.g. local mock testing with TEAM_TOKEN=test_token).
export TEAM_NAME="${TEAM_NAME:-AAA}"
export TEAM_TOKEN="${TEAM_TOKEN:-339cdbf6b25aece055e0f858efc89ccb}"

# Per 提交规范.txt §4: register the team at the top of run.sh, before
# vLLM startup consumes the first ~30s. client.py still registers again
# as a safety net — /register is idempotent.
if [ -n "$PLATFORM_URL" ]; then
    echo "[run] Registering team $TEAM_NAME at $PLATFORM_URL ..."
    REG_BODY="{\"name\":\"$TEAM_NAME\",\"token\":\"$TEAM_TOKEN\"}"
    REG_RESP=$(curl -sS -m 10 -X POST \
        -H "Content-Type: application/json" \
        -d "$REG_BODY" \
        "$PLATFORM_URL/register" 2>&1) && \
        echo "[run] Register response: $REG_RESP" || \
        echo "[run] WARN: early register failed (client.py will retry): $REG_RESP"
else
    echo "[run] PLATFORM_URL not set, skipping early register (client.py will register after vLLM ready)"
fi

MODEL=${MODEL_PATH:-"/mnt/model/Qwen3-32B"}
MODEL_NAME=$(basename "$MODEL")
VLLM_PORT=8000
export VLLM_PORT

# Detect bundled draft model for speculative decoding.
DRAFT_DIR="$(pwd)/draft_model"
HAS_DRAFT="no"
if [ -z "$SPEC_MODEL" ] && [ -d "$DRAFT_DIR" ]; then
    SPEC_MODEL="$DRAFT_DIR"
fi
if [ -n "$SPEC_MODEL" ] && [ -d "$SPEC_MODEL" ]; then
    HAS_DRAFT="yes"
    echo "[run] Draft model available: $SPEC_MODEL"
else
    echo "[run] No draft model found; speculative decoding will not be attempted"
fi

# Base vLLM args (shared across all fallback levels).
BASE_ARGS=(
    --model "$MODEL"
    --served-model-name "$MODEL_NAME"
    --tensor-parallel-size 4
    --port "$VLLM_PORT"
    --enable-prefix-caching
    --enable-chunked-prefill
    --max-num-seqs 16
    --max-num-batched-tokens 8192
    --gpu-memory-utilization 0.95
    --max-model-len 8192
    --disable-log-requests
    --disable-log-stats
)

VLLM_PID=""

# Wait up to $1 seconds for vLLM /v1/models to respond. Returns 0 on ready,
# 1 if the process died (flag error / OOM), 2 on timeout (model load too slow).
wait_for_vllm() {
    local timeout=$1
    local i
    for i in $(seq 1 "$timeout"); do
        if curl -s "http://localhost:${VLLM_PORT}/v1/models" > /dev/null 2>&1; then
            echo "[run] vLLM ready after ${i}s"
            return 0
        fi
        if ! kill -0 "$VLLM_PID" 2>/dev/null; then
            echo "[run] vLLM process died during startup (flag error or OOM)"
            return 1
        fi
        sleep 1
    done
    echo "[run] vLLM did not become ready within ${timeout}s"
    return 2
}

# Kill current vLLM instance and wait for port to free.
kill_vllm() {
    if [ -n "$VLLM_PID" ]; then
        kill -9 "$VLLM_PID" 2>/dev/null
        wait "$VLLM_PID" 2>/dev/null
        VLLM_PID=""
    fi
    # Give the OS a moment to release the port.
    for _ in $(seq 1 10); do
        if ! curl -s "http://localhost:${VLLM_PORT}/v1/models" > /dev/null 2>&1; then
            break
        fi
        sleep 1
    done
}

# Launch vLLM with extra args. Returns success only if the server accepts
# /v1/models within the timeout.
try_launch() {
    local desc="$1"; shift
    local timeout="$1"; shift
    echo "=============================================================="
    echo "[run] Attempting: $desc"
    echo "[run] Extra args: $*"
    echo "=============================================================="
    python -m vllm.entrypoints.openai.api_server "${BASE_ARGS[@]}" "$@" &
    VLLM_PID=$!
    wait_for_vllm "$timeout"
    local rc=$?
    if [ $rc -eq 0 ]; then
        echo "[run] SUCCESS: $desc"
        return 0
    fi
    echo "[run] FAILED: $desc (rc=$rc), cleaning up"
    kill_vllm
    return 1
}

# ----------------------------------------------------------------------------
# Fallback ladder (aggressive → conservative). Tries each until one works.
# Override any step by setting QUANT_MODE / SPEC_DISABLE env vars before run.
# ----------------------------------------------------------------------------
QUANT_MODE=${QUANT_MODE:-auto}    # auto | fp8 | bf16 | awq
SPEC_DISABLE=${SPEC_DISABLE:-0}   # 1 to skip all spec-decode attempts
NUM_SPEC_TOKENS=${NUM_SPEC_TOKENS:-5}

SUCCESS_DESC=""

# Build list of attempts based on mode and draft availability.
attempt() {
    local desc="$1"; shift
    if try_launch "$desc" 300 "$@"; then
        SUCCESS_DESC="$desc"
        return 0
    fi
    return 1
}

try_all() {
    # Level 1: FP8 + spec decode (fastest)
    if [ "$QUANT_MODE" = "auto" ] || [ "$QUANT_MODE" = "fp8" ]; then
        if [ "$HAS_DRAFT" = "yes" ] && [ "$SPEC_DISABLE" != "1" ]; then
            attempt "L1: FP8 + spec decode" \
                --quantization fp8 --kv-cache-dtype fp8 \
                --speculative-model "$SPEC_MODEL" \
                --num-speculative-tokens "$NUM_SPEC_TOKENS" && return 0
        fi
        # Level 1b: FP8 without spec
        attempt "L1b: FP8 only" \
            --quantization fp8 --kv-cache-dtype fp8 && return 0
    fi

    # Level 2: AWQ (only if user explicitly set QUANT_MODE=awq)
    if [ "$QUANT_MODE" = "awq" ]; then
        if [ -z "$AWQ_MODEL_PATH" ] || [ ! -d "$AWQ_MODEL_PATH" ]; then
            echo "[run] QUANT_MODE=awq but AWQ_MODEL_PATH not set/missing — skipping"
        else
            local awq_name
            awq_name=$(basename "$AWQ_MODEL_PATH")
            local orig_model="${BASE_ARGS[1]}"
            local orig_name="${BASE_ARGS[3]}"
            BASE_ARGS[1]="$AWQ_MODEL_PATH"
            BASE_ARGS[3]="$awq_name"
            if attempt "L2: AWQ" --quantization awq_marlin; then
                return 0
            fi
            # Restore original model path for subsequent fallbacks.
            BASE_ARGS[1]="$orig_model"
            BASE_ARGS[3]="$orig_name"
        fi
    fi

    # Level 3: BF16 + spec decode (safest speedup)
    if [ "$QUANT_MODE" != "bf16_baseline" ]; then
        if [ "$HAS_DRAFT" = "yes" ] && [ "$SPEC_DISABLE" != "1" ]; then
            attempt "L3: BF16 + spec decode" \
                --speculative-model "$SPEC_MODEL" \
                --num-speculative-tokens "$NUM_SPEC_TOKENS" && return 0
        fi
    fi

    # Level 4: BF16 baseline (bulletproof)
    attempt "L4: BF16 baseline" && return 0

    return 1
}

if ! try_all; then
    echo "[run] FATAL: all vLLM configurations failed to start"
    exit 1
fi
echo "[run] Proceeding with: $SUCCESS_DESC"

# ----------------------------------------------------------------------------
# Warmup — trigger CUDA graph capture with a few dummy requests.
# Non-fatal: a warmup failure doesn't stop the client from starting.
# ----------------------------------------------------------------------------
echo "[run] Warming up vLLM..."
python warmup.py --model "$MODEL_NAME" --port "$VLLM_PORT" || \
    echo "[run] WARN: warmup failed, continuing anyway"

# ----------------------------------------------------------------------------
# Client
# ----------------------------------------------------------------------------
echo "[run] Starting client..."
python client.py &
CLIENT_PID=$!

cleanup() {
    echo "[run] Shutting down..."
    kill "$CLIENT_PID" 2>/dev/null || true
    kill "$VLLM_PID" 2>/dev/null || true
    wait
}
trap cleanup EXIT INT TERM

wait "$CLIENT_PID"
