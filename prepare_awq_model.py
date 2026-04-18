"""
Download Qwen3-32B AWQ INT4 weights as a fallback quantization target.

This is the MORE aggressive quantization path (AWQ INT4 vs FP8). Only bother
running this if real-machine testing shows FP8 is not fast enough to hit
your target SLA tier.

Size: ~16-18 GB. At 13 MB/s (ModelScope from China) this takes ~20 minutes.
HF is usually slower from China — prefer --source modelscope.

Usage:
  python prepare_awq_model.py                         # default: ModelScope
  python prepare_awq_model.py --source hf             # if HF is fast for you
  python prepare_awq_model.py --dest D:\qwen3_awq     # download elsewhere

After download, enable AWQ in run.sh by setting env vars:
  QUANT_MODE=awq AWQ_MODEL_PATH=/path/to/qwen3_awq bash run.sh

The platform probably will NOT have Qwen3-32B-AWQ pre-mounted, so you need
to either (a) bundle it in submission.tar.gz (16+GB — likely over submission
size limit), or (b) download in setup.sh (eats into the 20min setup budget —
ModelScope download at 13MB/s takes ~20min, cutting it close).

Preferred stack for real run (rough order of effort vs payoff):
  1. FP8 (default in run.sh) — zero extra work, ~1.5-2x.   ← try this first
  2. AWQ INT4 — this script + config change. ~2-3x.       ← only if FP8 insufficient
  3. FP8 + speculative decoding — best of both worlds.    ← stack them.
"""

import argparse
import os
import shutil
import sys

DEFAULT_DEST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qwen3_32b_awq")


def download_hf(repo_id: str, dest: str):
    from huggingface_hub import snapshot_download
    print(f"[prepare-awq] HF: downloading {repo_id} → {dest}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=dest,
        allow_patterns=["*.json", "*.safetensors", "*.txt", "tokenizer*", "*.model"],
    )


def download_modelscope(repo_id: str, dest: str):
    from modelscope import snapshot_download as ms_download
    print(f"[prepare-awq] ModelScope: downloading {repo_id} (cache then copy → {dest})")
    tmp = ms_download(model_id=repo_id)
    if tmp != dest:
        os.makedirs(dest, exist_ok=True)
        for name in os.listdir(tmp):
            src = os.path.join(tmp, name)
            dst = os.path.join(dest, name)
            if os.path.isfile(src) and not os.path.exists(dst):
                shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="Qwen/Qwen3-32B-AWQ",
                    help="HF/ModelScope repo id")
    ap.add_argument("--dest", default=DEFAULT_DEST)
    ap.add_argument("--source", choices=["hf", "modelscope"], default="modelscope")
    args = ap.parse_args()

    os.makedirs(args.dest, exist_ok=True)

    try:
        if args.source == "hf":
            download_hf(args.repo, args.dest)
        else:
            download_modelscope(args.repo, args.dest)
    except Exception as e:
        print(f"[prepare-awq] download failed: {e}")
        print(f"[prepare-awq] Tip: check that the repo exists — some Qwen3 sizes "
              f"may not have an AWQ release yet.")
        return 1

    total = sum(
        os.path.getsize(os.path.join(args.dest, f))
        for f in os.listdir(args.dest) if os.path.isfile(os.path.join(args.dest, f))
    ) / (1024 ** 3)
    print(f"[prepare-awq] done. Size: {total:.2f} GB at {args.dest}")
    print(f"[prepare-awq] To use: QUANT_MODE=awq AWQ_MODEL_PATH={args.dest} bash run.sh")
    return 0


if __name__ == "__main__":
    sys.exit(main())
