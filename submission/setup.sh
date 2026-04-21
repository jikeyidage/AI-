#!/bin/bash
set -e

cd "$(dirname "$0")"

# -----------------------------------------------------------------------------
# Step 1: Ensure torch + CUDA are available.
#   Fast path: platform has Blackwell-compatible torch pre-installed → keep it.
#   Fallback : bootstrap torch 2.7.0 + cu128 family from pytorch.org (needs net).
# Rationale: 4x RTX 5090 is Blackwell sm_120, needs CUDA 12.8 runtime.
# PyPI's default torch==2.7.0 wheel is cu126 (no sm_120 kernels) — we must use
# the pytorch.org cu128 index when bootstrapping.
# -----------------------------------------------------------------------------
echo "[setup] Step 1: Ensuring torch + CUDA available..."
if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    python -c "
import torch
print(f'[setup] Platform torch detected: {torch.__version__}  cuda={torch.version.cuda}  '
      f'devices={torch.cuda.device_count()}')
"
else
    echo "[setup] No working torch+CUDA on platform; bootstrapping torch 2.7.0 + cu128..."
    pip install --index-url https://download.pytorch.org/whl/cu128 \
        torch==2.7.0 torchaudio==2.7.0 torchvision==0.22.0 \
        || { echo "[setup] FATAL: torch cu128 bootstrap failed (no network?)"; exit 1; }
    python -c "
import torch
assert torch.cuda.is_available(), 'CUDA unavailable even after install'
print(f'[setup] Bootstrapped torch {torch.__version__}  cuda={torch.version.cuda}  '
      f'devices={torch.cuda.device_count()}')
" || { echo "[setup] FATAL: torch installed but CUDA still unavailable"; exit 1; }
fi

# -----------------------------------------------------------------------------
# Step 2: Install or verify vllm.
#   - If platform already has vllm 0.9+ / 0.10+ / 0.11+, keep it (its deps are
#     already there; reinstalling risks binary/ABI drift).
#   - Otherwise install vllm==0.9.2.
#       - If torch was platform-pre-installed (Step 1 fast path), use --no-deps
#         to avoid any chance of pip pulling PyPI's cu126 torch over the
#         Blackwell wheel. vLLM's other runtime deps are assumed already present
#         (the platform shipped vLLM, so its deps shipped with it).
#       - If we bootstrapped torch (Step 1 fallback path), do a full install so
#         xformers / ray / transformers / etc. get pulled in. PEP 440 says
#         `torch==2.7.0` matches `torch==2.7.0+cu128`, so the already-installed
#         Blackwell torch satisfies vLLM's hard requirement without being
#         replaced.
# -----------------------------------------------------------------------------
echo "[setup] Step 2: Ensuring vllm installed..."
CURRENT_VLLM=$(python -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo "none")
echo "[setup] Current vllm: $CURRENT_VLLM"

# $BOOTSTRAPPED is "yes" if Step 1 took the fallback path.
# We detect by checking whether transformers is also installed — if not, we're
# on a greenfield environment and need vllm's full dep tree.
HAS_VLLM_DEPS=$(python -c "import transformers, tokenizers; print('yes')" 2>/dev/null || echo "no")

case "$CURRENT_VLLM" in
    0.9.*|0.10.*|0.11.*)
        echo "[setup] vllm $CURRENT_VLLM is compatible (0.9+ CLI syntax). Keeping it."
        ;;
    *)
        if [ "$HAS_VLLM_DEPS" = "yes" ]; then
            echo "[setup] Installing vllm==0.9.2 (--no-deps; platform has deps)..."
            pip install --no-deps vllm==0.9.2
        else
            echo "[setup] Installing vllm==0.9.2 (full deps; greenfield env)..."
            # Normal install — torch is already satisfied (local +cu128 tag matches
            # ==2.7.0 per PEP 440), so pip will fetch xformers/ray/etc. without
            # touching torch.
            pip install vllm==0.9.2
        fi
        ;;
esac

# -----------------------------------------------------------------------------
# Step 3: Install our own extra — httpx (used by client.py + warmup.py).
# Pure Python, safe to pin exactly.
# -----------------------------------------------------------------------------
echo "[setup] Step 3: Installing httpx==0.27.2..."
pip install httpx==0.27.2

# -----------------------------------------------------------------------------
# Step 4: Import sanity check — all critical packages load before run.sh starts
# spinning up vLLM. Catches missing transitive deps early (cheaper to fail here
# than after vLLM eats 30s of GPU memory).
# -----------------------------------------------------------------------------
echo "[setup] Step 4: Import sanity check..."
python -c "
import vllm, httpx, transformers, torch
print(f'[setup] vllm={vllm.__version__}  torch={torch.__version__}  '
      f'transformers={transformers.__version__}  httpx={httpx.__version__}')
" || { echo "[setup] FATAL: import check failed"; exit 1; }

# -----------------------------------------------------------------------------
# Step 5 (optional): download draft model for speculative decoding.
# Disabled by default — draft model is typically bundled inside
# submission/draft_model/ at build time. Enable via:
#   DOWNLOAD_SPEC_MODEL=1 bash setup.sh
# -----------------------------------------------------------------------------
if [ "${DOWNLOAD_SPEC_MODEL:-0}" = "1" ]; then
    DRAFT_DIR=${SPEC_MODEL:-/tmp/spec_draft}
    DRAFT_REPO=${SPEC_REPO:-Qwen/Qwen3-1.5B}
    echo "[setup] Step 5: Downloading draft model ${DRAFT_REPO} → ${DRAFT_DIR}"
    pip install --quiet huggingface_hub
    python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='${DRAFT_REPO}', local_dir='${DRAFT_DIR}', local_dir_use_symlinks=False)
print('[setup] draft model ready at ${DRAFT_DIR}')
" || echo "[setup] WARN: draft download failed; speculative decoding will be disabled"
fi

echo "[setup] Done."
