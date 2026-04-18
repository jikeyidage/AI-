"""
Windows-friendly launcher for ubiservice.

bin/start.py shells out to `redis-server` / `redis-cli`, which don't exist
on Windows when Memurai is used (Memurai ships `memurai.exe` /
`memurai-cli.exe`). Since Memurai already runs as a Windows service on
port 6379, this launcher skips Redis process management entirely and only
starts:
  - the background task generator thread
  - the FastAPI matcher on port 8003

Ctrl+C to stop.
"""

import json
import os
import signal
import subprocess
import sys
import threading
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UBISERVICE_DIR = os.path.join(BASE_DIR, "ubiservice")
SRC_DIR = os.path.join(UBISERVICE_DIR, "src")
CONFIG_DIR = os.path.join(UBISERVICE_DIR, "config")
sys.path.insert(0, SRC_DIR)

from defination import ConfigRegistry
from task_builder import TaskBuilder
from fake_generator import generate_messages

REDIS_PORT = 6379
MATCHER_PORT = 8003
CONFIG_PATH = os.path.join(CONFIG_DIR, "defination_base.json")

_matcher_proc = None
_stop_event = threading.Event()


def check_redis():
    """Ping Redis/Memurai via the Python client."""
    import redis as sync_redis
    try:
        rdb = sync_redis.Redis(host="localhost", port=REDIS_PORT, socket_timeout=2)
        pong = rdb.ping()
        rdb.close()
        if pong:
            print(f"[start] Redis/Memurai reachable on port {REDIS_PORT}")
            return True
    except Exception as e:
        print(f"[start] ERROR: cannot reach Redis on {REDIS_PORT}: {e}")
        return False
    return False


def task_generator_loop():
    """Mirror of ubiservice/bin/start.py task_generator_loop."""
    import redis as sync_redis

    rdb = sync_redis.Redis(host="localhost", port=REDIS_PORT, decode_responses=False)

    # Wipe any leftover state from previous runs so the queue starts clean.
    try:
        rdb.flushdb()
    except Exception as e:
        print(f"[start] WARN: flushdb failed: {e}")

    registry = ConfigRegistry(CONFIG_PATH)
    builder = TaskBuilder(rank=0, registry=registry, max_wait_time=0.3)
    gen = generate_messages(registry, msg_id_start=0)

    task_count = 0
    print("[start] Task generator started")

    while not _stop_event.is_set():
        num_buckets = len(builder.buckets)
        for _ in range(num_buckets + 1):
            msg = next(gen)
            builder.put(msg)

        task = builder.maybe_build()
        if task is not None:
            task_id = task.overview.task_id
            task_data = task.model_dump()
            pipe = rdb.pipeline()
            pipe.hset(
                f"task:{task_id}",
                mapping={
                    "overview": json.dumps(task.overview.model_dump()),
                    "full": json.dumps(task_data),
                    "max_winners": task.overview.max_winners,
                },
            )
            pipe.rpush("task_queue", task_id)
            pipe.execute()
            task_count += 1
            if task_count % 50 == 0:
                print(f"[start] Generated {task_count} tasks so far")

        time.sleep(0.1)

    rdb.close()
    print(f"[start] Task generator stopped ({task_count} tasks total)")


def start_matcher():
    global _matcher_proc
    env = os.environ.copy()
    env["REDIS_URL"] = f"redis://localhost:{REDIS_PORT}"
    _matcher_proc = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn",
            "matcher:app",
            "--host", "0.0.0.0",
            "--port", str(MATCHER_PORT),
            "--log-level", "info",
        ],
        cwd=SRC_DIR,
        env=env,
    )
    print(f"[start] Matcher starting on port {MATCHER_PORT}")


def cleanup(*args):
    print("\n[start] Shutting down...")
    _stop_event.set()
    if _matcher_proc and _matcher_proc.poll() is None:
        _matcher_proc.terminate()
        try:
            _matcher_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _matcher_proc.kill()
        print("[start] Matcher stopped")
    print("[start] Bye!")
    sys.exit(0)


def main():
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    print("=" * 60)
    print("  ubiservice launcher (Windows / Memurai)")
    print("=" * 60)

    if not check_redis():
        print("[start] Is Memurai running? Try: Get-Service Memurai")
        sys.exit(1)

    gen_thread = threading.Thread(target=task_generator_loop, daemon=True)
    gen_thread.start()
    time.sleep(1.0)

    start_matcher()
    time.sleep(1.5)

    print()
    print("=" * 60)
    print(f"  Mock server ready!  http://127.0.0.1:{MATCHER_PORT}")
    print("  Press Ctrl+C to stop")
    print("=" * 60)

    try:
        while True:
            if _matcher_proc and _matcher_proc.poll() is not None:
                print("[start] WARNING: Matcher exited unexpectedly")
                cleanup()
            time.sleep(2)
    except KeyboardInterrupt:
        cleanup()


if __name__ == "__main__":
    main()
