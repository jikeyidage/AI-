"""
Download a Qwen3 draft model and place it at submission/draft_model/.

Run this ONCE before final packaging (on a machine with internet and enough
disk). The downloaded weights will be bundled into submission.tar.gz by
run_eval_windows.py if --include-draft is passed.

Rationale:
  Q33 in the competition FAQ explicitly allows bringing your own draft model.
  The platform is NOT guaranteed to have a small Qwen3 variant on disk, so
  shipping it with the submission is the only reliable route.

Size reference (BF16):
  Qwen3-0.6B  ~1.2 GB
  Qwen3-1.7B  ~3 GB       (recommended — good accept rate vs size tradeoff)
  Qwen3-4B    ~8 GB       (may be too large to bundle)

Usage:
  python prepare_draft_model.py                          # default Qwen3-1.7B
  python prepare_draft_model.py --repo Qwen/Qwen3-0.6B
  python prepare_draft_model.py --repo Qwen/Qwen3-1.7B --source modelscope
"""

import argparse
import os
import shutil
import sys

DEFAULT_DEST = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "submission", "draft_model")


def download_hf(repo_id: str, dest: str):
    from huggingface_hub import snapshot_download
    print(f"[prepare] HF: downloading {repo_id} → {dest}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=dest,
        local_dir_use_symlinks=False,
        allow_patterns=["*.json", "*.safetensors", "*.txt", "tokenizer*", "*.model"],
    )


def download_modelscope(repo_id: str, dest: str):
    from modelscope import snapshot_download as ms_download
    print(f"[prepare] ModelScope: downloading {repo_id} → {dest}")
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
    ap.add_argument("--repo", default="Qwen/Qwen3-1.7B",
                    help="HF/ModelScope repo id")
    ap.add_argument("--dest", default=DEFAULT_DEST,
                    help="Destination directory (must be inside submission/ to auto-detect)")
    ap.add_argument("--source", choices=["hf", "modelscope"], default="hf",
                    help="Download source (hf default; modelscope helps if HF is slow)")
    args = ap.parse_args()

    os.makedirs(args.dest, exist_ok=True)

    if args.source == "hf":
        try:
            download_hf(args.repo, args.dest)
        except Exception as e:
            print(f"[prepare] HF download failed: {e}")
            print("[prepare] Tip: retry with --source modelscope")
            return 1
    else:
        try:
            download_modelscope(args.repo, args.dest)
        except Exception as e:
            print(f"[prepare] ModelScope download failed: {e}")
            return 1

    # Sanity check
    expected = ["config.json"]
    missing = [f for f in expected if not os.path.exists(os.path.join(args.dest, f))]
    if missing:
        print(f"[prepare] WARN: expected files missing: {missing}")
    size_mb = sum(
        os.path.getsize(os.path.join(args.dest, f))
        for f in os.listdir(args.dest) if os.path.isfile(os.path.join(args.dest, f))
    ) / (1024 * 1024)
    print(f"[prepare] done. Total size: {size_mb:.1f} MB at {args.dest}")
    print("[prepare] Next: run `python run_eval_windows.py --include-draft` to test,")
    print("[prepare]        or package with `tar czf submission.tar.gz -C submission .`")
    return 0


if __name__ == "__main__":
    sys.exit(main())
