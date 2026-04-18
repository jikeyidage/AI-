"""
Windows-native equivalent of ubiservice/bin/run_eval.py.

Does what run_eval.py does, minus the Unix-only bits (os.setsid, os.killpg,
bash) and minus the actual vLLM startup (we don't have a GPU). Flow:

  1. Package submission/ → submission.tar.gz (validates our packaging)
  2. Extract to a temp dir (validates platform-side extraction)
  3. Write contest.json matching run_eval.py's format
  4. Start ubiservice (matcher + task generator) via start_ubiservice_local.py
  5. Start mock_vllm (replaces the real vLLM server that run.sh would start)
  6. Start submission/client.py with run_eval-style env vars (CONFIG_PATH,
     PLATFORM_URL, TEAM_NAME, TEAM_TOKEN, MODEL_PATH)
  7. Run for --duration seconds
  8. Fetch leaderboard → write result JSON

This validates: tarball structure, env var handling, contest.json parsing,
path resolution in client.py, and steady-state throughput.
"""

import argparse
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import time
import urllib.request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SUBMISSION_DIR = os.path.join(BASE_DIR, "submission")
UBISERVICE_DIR = os.path.join(BASE_DIR, "ubiservice")
CONFIG_DIR = os.path.join(UBISERVICE_DIR, "config")
PLATFORM_PORT = 8003
VLLM_PORT = 8000


_PACK_EXCLUDE = {"__pycache__", ".pytest_cache", ".DS_Store"}


def _make_pack_filter(include_draft: bool):
    def _pack_filter(tarinfo: tarfile.TarInfo):
        # tarinfo.name is relative to the archive root ("draft_model/config.json")
        parts = tarinfo.name.split("/")
        top = parts[0] if parts else ""
        if top == "draft_model" and not include_draft:
            return None
        name = os.path.basename(tarinfo.name.rstrip("/"))
        if name in _PACK_EXCLUDE or name.endswith(".pyc"):
            return None
        return tarinfo
    return _pack_filter


def package_submission(dest_tar: str, include_draft: bool = False):
    """Create a .tar.gz of submission/ contents (NOT the submission/ dir itself)."""
    if os.path.exists(dest_tar):
        os.remove(dest_tar)
    flt = _make_pack_filter(include_draft)
    with tarfile.open(dest_tar, "w:gz") as tf:
        for name in os.listdir(SUBMISSION_DIR):
            if name in _PACK_EXCLUDE:
                continue
            if name == "draft_model" and not include_draft:
                continue
            tf.add(os.path.join(SUBMISSION_DIR, name), arcname=name, filter=flt)
    size_mb = os.path.getsize(dest_tar) / (1024 * 1024)
    size_kb = os.path.getsize(dest_tar) / 1024
    size_str = f"{size_mb:.1f} MB" if size_mb >= 1 else f"{size_kb:.1f} KB"
    print(f"[pack] {dest_tar}  ({size_str})  include_draft={include_draft}")


def extract_submission(archive: str, dest_dir: str):
    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir)
    os.makedirs(dest_dir, exist_ok=True)
    with tarfile.open(archive) as tf:
        tf.extractall(dest_dir)
    contents = sorted(os.listdir(dest_dir))
    print(f"[extract] → {dest_dir}  contents: {contents}")
    for required in ("run.sh", "setup.sh", "client.py"):
        assert required in contents, f"missing {required} in archive"


def write_contest_config(dest_json: str, platform_url: str, duration: int, model_path: str):
    defination_path = os.path.join(CONFIG_DIR, "defination_base.json")
    with open(defination_path) as f:
        defn = json.load(f)
    sla_levels = {n: {"latency_max": spec.get("ttft_avg", 10.0)}
                  for n, spec in defn.get("SLA", {}).items()}
    config = {
        "platform_url": platform_url,
        "model_name": "Qwen3-32B",
        "model_path": model_path,
        "contestant_port": 9000,
        "duration_s": duration,
        "sla_levels": sla_levels,
        "sampling_params": defn.get("SamplingParam", {}),
    }
    with open(dest_json, "w") as f:
        json.dump(config, f, indent=2)
    print(f"[config] {dest_json}")


def wait_for_port(host: str, port: int, timeout: float) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except OSError:
            time.sleep(0.3)
    return False


def fetch_leaderboard(platform_url: str, team: str):
    try:
        with urllib.request.urlopen(f"{platform_url}/leaderboard", timeout=5) as r:
            lb = json.loads(r.read())
        for e in lb.get("leaderboard", []):
            if e.get("name") == team:
                return e
    except Exception as ex:
        print(f"[leaderboard] error: {ex}")
    return None


def kill_proc(p: subprocess.Popen):
    if p and p.poll() is None:
        try:
            p.terminate()
            p.wait(timeout=5)
        except Exception:
            try:
                p.kill()
            except Exception:
                pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=int, default=30)
    ap.add_argument("--team-name", default="test_team")
    ap.add_argument("--team-token", default="test_token")
    ap.add_argument("--work-dir", default=os.path.join(BASE_DIR, "_eval_work"))
    ap.add_argument("--output", default=os.path.join(BASE_DIR, "_eval_work", "result.json"))
    ap.add_argument("--conda-env", default="inference")
    ap.add_argument("--python",
                    default=r"C:\Users\Keyi\miniconda3\envs\inference\python.exe",
                    help="Python executable to use (defaults to inference env)")
    ap.add_argument("--mock-latency-ms", type=int, default=20,
                    help="Per-request latency to simulate in mock_vllm")
    ap.add_argument("--include-draft", action="store_true",
                    help="Include submission/draft_model/ in the tarball (for real-run packaging)")
    args = ap.parse_args()

    platform_url = f"http://127.0.0.1:{PLATFORM_PORT}"
    work_dir = os.path.abspath(args.work_dir)
    os.makedirs(work_dir, exist_ok=True)

    archive = os.path.join(work_dir, "submission.tar.gz")
    ext_dir = os.path.join(work_dir, "submission")
    cfg = os.path.join(work_dir, "contest.json")

    result = {
        "team": args.team_name,
        "duration_s": args.duration,
        "pack_ok": False,
        "extract_ok": False,
        "platform_ok": False,
        "mock_vllm_ok": False,
        "client_ok": False,
        "score": 0.0,
        "tasks_completed": 0,
        "tasks_accepted": 0,
        "avg_latency_ms": 0.0,
        "error": None,
    }

    # Phase 1: pack + extract
    print("\n=== Phase 1: pack + extract ===")
    try:
        package_submission(archive, include_draft=args.include_draft)
        result["pack_ok"] = True
        extract_submission(archive, ext_dir)
        result["extract_ok"] = True
        write_contest_config(cfg, platform_url, args.duration, "/mnt/model/Qwen3-32B")
    except Exception as ex:
        result["error"] = f"pack/extract failed: {ex}"
        print(result["error"])
        _write_result(result, args.output)
        return 1

    platform_proc = None
    mock_proc = None
    client_proc = None

    try:
        # Phase 2: platform
        print("\n=== Phase 2: start ubiservice ===")
        platform_proc = subprocess.Popen(
            [args.python, "start_ubiservice_local.py"],
            cwd=BASE_DIR,
            stdout=open(os.path.join(work_dir, "platform.log"), "w"),
            stderr=subprocess.STDOUT,
        )
        if not wait_for_port("127.0.0.1", PLATFORM_PORT, timeout=30):
            raise RuntimeError("ubiservice did not become ready in 30s")
        time.sleep(1.5)  # let task generator produce initial tasks
        result["platform_ok"] = True
        print(f"[platform] ready on {platform_url}")

        # Phase 3: mock vLLM
        print("\n=== Phase 3: start mock vLLM ===")
        mock_proc = subprocess.Popen(
            [args.python, "mock_vllm.py", "--port", str(VLLM_PORT),
             "--model-name", "Qwen3-32B", "--latency-ms", str(args.mock_latency_ms)],
            cwd=BASE_DIR,
            stdout=open(os.path.join(work_dir, "mock_vllm.log"), "w"),
            stderr=subprocess.STDOUT,
        )
        if not wait_for_port("127.0.0.1", VLLM_PORT, timeout=15):
            raise RuntimeError("mock_vllm did not become ready in 15s")
        result["mock_vllm_ok"] = True
        print(f"[mock_vllm] ready on :{VLLM_PORT}")

        # Phase 4: client (from EXTRACTED submission dir, using run_eval env vars)
        print("\n=== Phase 4: start client ===")
        env = os.environ.copy()
        env["CONFIG_PATH"] = cfg
        env["PLATFORM_URL"] = platform_url
        env["TEAM_NAME"] = args.team_name
        env["TEAM_TOKEN"] = args.team_token
        env["MODEL_PATH"] = "Qwen3-32B"  # force fallback tokenizer (no real weights)
        env["VLLM_PORT"] = str(VLLM_PORT)
        client_proc = subprocess.Popen(
            [args.python, "client.py"],
            cwd=ext_dir,
            env=env,
            stdout=open(os.path.join(work_dir, "client.log"), "w"),
            stderr=subprocess.STDOUT,
        )
        result["client_ok"] = True
        time.sleep(3.0)

        # Phase 5: run + collect
        print(f"\n=== Phase 5: run for {args.duration}s ===")
        start = time.time()
        while time.time() - start < args.duration:
            time.sleep(min(10, args.duration - (time.time() - start)))
            entry = fetch_leaderboard(platform_url, args.team_name)
            elapsed = time.time() - start
            if entry:
                print(f"  [{elapsed:5.1f}s] score={entry['score']:>6.1f}  "
                      f"completed={entry['tasks_completed']:>4}  "
                      f"accepted={entry['tasks_accepted']:>4}  "
                      f"latency={entry.get('avg_latency_ms', 0):>5.1f}ms")
            else:
                print(f"  [{elapsed:5.1f}s] team not on leaderboard yet")

        entry = fetch_leaderboard(platform_url, args.team_name)
        if entry:
            result["score"] = entry.get("score", 0.0)
            result["tasks_completed"] = entry.get("tasks_completed", 0)
            result["tasks_accepted"] = entry.get("tasks_accepted", 0)
            result["avg_latency_ms"] = entry.get("avg_latency_ms", 0.0)

        # Count LATE vs OK in client log — sanity check for SLA stress runs
        try:
            with open(os.path.join(work_dir, "client.log"), encoding="utf-8", errors="replace") as f:
                log = f.read()
            result["sla_ok"] = log.count(", OK)")
            result["sla_late"] = log.count(", LATE)")
        except Exception:
            pass

    except Exception as ex:
        result["error"] = str(ex)
        print(f"[error] {ex}")

    finally:
        print("\n=== Teardown ===")
        for name, p in [("client", client_proc), ("mock_vllm", mock_proc), ("platform", platform_proc)]:
            kill_proc(p)
            print(f"  stopped {name}")

    _write_result(result, args.output)
    print("\n" + "=" * 60)
    print(f"  Eval result (team={args.team_name}, duration={args.duration}s)")
    print("=" * 60)
    for k, v in result.items():
        print(f"  {k:20s} = {v}")
    print("=" * 60)
    return 0 if result["score"] > 0 else 1


def _write_result(result: dict, path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[result] {path}")


if __name__ == "__main__":
    sys.exit(main())
