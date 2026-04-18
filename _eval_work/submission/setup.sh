#!/bin/bash
set -e

cd "$(dirname "$0")"

echo "[setup] Installing Python dependencies..."
pip install -r requirements.txt
echo "[setup] Python deps done."

# --- Optional: download a draft model for speculative decoding ---------------
# Tier 2 optimization. Disabled by default because:
#   - setup has a 20min budget; HF download of a 3-7B model can eat ~5-10min
#   - the draft model path might be pre-mounted on the platform (check first)
#
# Enable by running: DOWNLOAD_SPEC_MODEL=1 bash setup.sh
# Or hard-code a path in run.sh by setting SPEC_MODEL=/path/to/draft.
if [ "${DOWNLOAD_SPEC_MODEL:-0}" = "1" ]; then
    DRAFT_DIR=${SPEC_MODEL:-/tmp/spec_draft}
    DRAFT_REPO=${SPEC_REPO:-Qwen/Qwen3-1.5B}
    echo "[setup] Downloading draft model ${DRAFT_REPO} → ${DRAFT_DIR}"
    pip install --quiet huggingface_hub
    python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='${DRAFT_REPO}', local_dir='${DRAFT_DIR}', local_dir_use_symlinks=False)
print('[setup] draft model ready at ${DRAFT_DIR}')
" || echo "[setup] WARN: draft download failed; speculative decoding will be disabled"
fi

echo "[setup] Done."
