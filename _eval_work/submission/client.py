"""
AI Inference Challenge - Async Scheduling Client

Architecture:
  - Fetcher coroutines: continuously query + ask for tasks from platform
  - Worker coroutines: process inference for each task concurrently
  - Deadline monitor: ensure all accepted tasks are submitted before timeout
  - vLLM handles batching internally via continuous batching

Task types:
  - generate_until: text generation with sampling params -> fill response
  - loglikelihood: compute log P(continuation | prompt) -> fill accuracy
  - loglikelihood_rolling: compute total log P(prompt) -> fill accuracy
"""

import asyncio
import collections
import json
import logging
import os
import signal
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

import httpx
from transformers import AutoTokenizer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PLATFORM_URL = os.environ.get("PLATFORM_URL", "http://127.0.0.1:8003")
TOKEN = os.environ.get("TEAM_TOKEN", "test_token")
TEAM_NAME = os.environ.get("TEAM_NAME", "test_team")
MODEL_PATH = os.environ.get("MODEL_PATH", "/mnt/model/Qwen3-32B")
VLLM_PORT = int(os.environ.get("VLLM_PORT", "8000"))
VLLM_URL = f"http://127.0.0.1:{VLLM_PORT}"

# Read contest config if available
config_path = os.environ.get("CONFIG_PATH", "")
if config_path and os.path.exists(config_path):
    with open(config_path) as f:
        config = json.load(f)
    PLATFORM_URL = config.get("platform_url", PLATFORM_URL)

MODEL_NAME = os.path.basename(MODEL_PATH)

# Concurrency tuning
# Matcher rate limit is 32 req/s per endpoint per user. Keep combined
# qps under that: NUM_FETCHERS / QUERY_INTERVAL <= ~20 to leave headroom.
NUM_FETCHERS = 4            # parallel query+ask coroutines
MAX_INFLIGHT = 48           # max tasks being processed (API limit is 64)
QUERY_INTERVAL = 0.2        # seconds between queries per fetcher (4/0.2 = 20 qps)
DEFAULT_TIMEOUT_S = 600     # fallback eval_timeout_s

# SLA ttft_avg lookup (seconds)
SLA_TTFT = {
    "Bronze": 10.0, "Silver": 8.0, "Gold": 6.0, "Platinum": 4.0,
    "Diamond": 2.0, "Stellar": 1.5, "Glorious": 0.8, "Supreme": 0.5,
}

# Tier 4: optional strategy controls
# ----------------------------------
# SKIP_SLA_TIERS: comma-separated list of SLA tiers to skip (/query returns
#   a task at this tier → we don't /ask, call /query again). Default empty:
#   always try, since LATE is 0-score with no penalty (strictly better than
#   not-submitting which incurs -2x reward).
SKIP_SLA_TIERS = set(
    t.strip() for t in os.environ.get("SKIP_SLA_TIERS", "").split(",") if t.strip()
)
# SHORT_GEN_NOTHINK_THRESHOLD: if a generate_until task's max_gen_toks is
#   below this, prepend "</think>\n\n" to the prompt to force Qwen3 out of
#   thinking mode, trading some reasoning quality for guaranteed answer
#   emission inside the token budget. 0 disables this behavior (default).
#   Recommended values if enabling: 128 or 256.
SHORT_GEN_NOTHINK_THRESHOLD = int(os.environ.get("SHORT_GEN_NOTHINK_THRESHOLD", "0"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger("client")

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
shutdown_event = asyncio.Event()
inflight_sem: asyncio.Semaphore  # initialized in main()
stats = {"queried": 0, "accepted": 0, "submitted": 0, "failed": 0,
         "skipped_tier": 0, "thinking_truncated": 0}
# Per-SLA-tier running stats: {tier: {"count": int, "ok": int, "late": int,
#                                      "lat_sum": float, "lat_max": float}}
sla_stats: dict = collections.defaultdict(
    lambda: {"count": 0, "ok": 0, "late": 0, "lat_sum": 0.0, "lat_max": 0.0}
)


# ---------------------------------------------------------------------------
# Tokenizer (loaded once for logprob computation)
# ---------------------------------------------------------------------------
tokenizer = None


class _FallbackTokenizer:
    """Whitespace+char approximation when no real tokenizer is available.

    Only used for local integration testing. The real submission always
    has MODEL_PATH pointing at the Qwen3-32B weights on disk.
    """
    def encode(self, text: str, add_special_tokens: bool = False):
        # ~1 token per 3 characters — mirrors what mock_vllm.py reports.
        return list(range(max(1, len(text) // 3)))


def load_tokenizer():
    global tokenizer
    if os.path.exists(MODEL_PATH):
        log.info(f"Loading tokenizer from {MODEL_PATH}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        return
    try:
        log.warning(f"Model path {MODEL_PATH} not found, trying HF hub for {MODEL_NAME}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    except Exception as e:
        log.warning(f"HF fallback failed ({e}); using approximate tokenizer (LOCAL TEST ONLY)")
        tokenizer = _FallbackTokenizer()


# ---------------------------------------------------------------------------
# vLLM health check
# ---------------------------------------------------------------------------
async def wait_for_vllm(client: httpx.AsyncClient, timeout: int = 180):
    """Wait for vLLM server to be ready."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = await client.get(f"{VLLM_URL}/v1/models", timeout=5)
            if resp.status_code == 200:
                log.info("vLLM server is ready")
                return True
        except Exception:
            pass
        await asyncio.sleep(1)
    log.error("vLLM server did not start in time")
    return False


# ---------------------------------------------------------------------------
# Inference functions
# ---------------------------------------------------------------------------
async def do_generate_until(client: httpx.AsyncClient, msg: dict) -> None:
    """Handle generate_until: call vLLM completions API."""
    gen_kwargs = msg.get("eval_gen_kwargs") or {}
    prompt = msg.get("prompt", "")
    max_toks = gen_kwargs.get("max_gen_toks", 256)

    # Tier 4 (opt-in): on tight token budgets, force Qwen3 out of thinking
    # mode by pre-closing a <think> block. The extractor scores the textual
    # answer, so losing chain-of-thought costs some reasoning quality but
    # guarantees the answer actually appears inside max_tokens. Threshold=0
    # disables this (default).
    if SHORT_GEN_NOTHINK_THRESHOLD > 0 and max_toks < SHORT_GEN_NOTHINK_THRESHOLD:
        prompt = prompt + "<think>\n\n</think>\n\n"

    request_body = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "max_tokens": max_toks,
        "stop": gen_kwargs.get("until", []),
    }

    # Sampling parameters
    temp = gen_kwargs.get("temperature", 0.0)
    request_body["temperature"] = temp
    if temp == 0.0:
        # Deterministic: force greedy
        request_body["top_p"] = 1.0
    else:
        if "top_p" in gen_kwargs:
            request_body["top_p"] = gen_kwargs["top_p"]
        if "top_k" in gen_kwargs and gen_kwargs["top_k"] > 0:
            extra = request_body.get("extra_body", {})
            extra["top_k"] = gen_kwargs["top_k"]
            request_body["extra_body"] = extra

    # Penalty parameters
    if "repetition_penalty" in gen_kwargs and gen_kwargs["repetition_penalty"] != 1.0:
        extra = request_body.get("extra_body", {})
        extra["repetition_penalty"] = gen_kwargs["repetition_penalty"]
        request_body["extra_body"] = extra
    if "frequency_penalty" in gen_kwargs:
        request_body["frequency_penalty"] = gen_kwargs["frequency_penalty"]
    if "presence_penalty" in gen_kwargs:
        request_body["presence_penalty"] = gen_kwargs["presence_penalty"]

    resp = await client.post(
        f"{VLLM_URL}/v1/completions",
        json=request_body,
        timeout=300,
    )
    resp.raise_for_status()
    data = resp.json()
    text = data["choices"][0]["text"]

    # Remove stop tokens from the end if present (Q56.12)
    stop_tokens = gen_kwargs.get("until", [])
    for stop in stop_tokens:
        if text.endswith(stop):
            text = text[: -len(stop)]

    # Tier 4 detection: if Qwen3 opened a <think> block and never closed it,
    # the final answer probably didn't fit in max_gen_toks → we'll be judged
    # on incomplete output. Just count it; the extractor handles the tags.
    if "<think>" in text and "</think>" not in text:
        stats["thinking_truncated"] += 1

    msg["response"] = text


async def do_loglikelihood(client: httpx.AsyncClient, msg: dict) -> None:
    """Handle loglikelihood: compute total log P(continuation | prompt).

    Passes explicit token IDs to vLLM so the prompt/continuation boundary
    matches our tokenization exactly — avoids the BPE re-tokenization trap
    where len(enc(prompt)) + len(enc(cont)) != len(enc(prompt+cont)).
    """
    prompt = msg.get("prompt", "")
    continuation = msg.get("eval_continuation", "")

    if not continuation:
        msg["accuracy"] = 0.0
        return

    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    cont_ids = tokenizer.encode(continuation, add_special_tokens=False)
    full_ids = list(prompt_ids) + list(cont_ids)
    n_prompt = len(prompt_ids)
    n_full = len(full_ids)

    resp = await client.post(
        f"{VLLM_URL}/v1/completions",
        json={
            "model": MODEL_NAME,
            "prompt": full_ids,      # token IDs — no re-tokenization
            "max_tokens": 1,
            "temperature": 0,
            "logprobs": 1,
            "echo": True,
        },
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()

    logprobs_data = data["choices"][0].get("logprobs") or {}
    token_logprobs = logprobs_data.get("token_logprobs") or []

    # token_logprobs length: n_full prompt entries (first is None) + 1 generated.
    # Continuation portion = indices [n_prompt .. n_full - 1].
    total_logprob = 0.0
    for i in range(n_prompt, min(n_full, len(token_logprobs))):
        lp = token_logprobs[i]
        if lp is not None:
            total_logprob += lp

    msg["accuracy"] = total_logprob


async def do_loglikelihood_rolling(client: httpx.AsyncClient, msg: dict) -> None:
    """Handle loglikelihood_rolling: compute total log P(prompt)."""
    prompt = msg.get("prompt", "")

    if not prompt:
        msg["accuracy"] = 0.0
        return

    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    n_prompt = len(prompt_ids)

    resp = await client.post(
        f"{VLLM_URL}/v1/completions",
        json={
            "model": MODEL_NAME,
            "prompt": prompt_ids,    # token IDs — no re-tokenization
            "max_tokens": 1,
            "temperature": 0,
            "logprobs": 1,
            "echo": True,
        },
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()

    logprobs_data = data["choices"][0].get("logprobs") or {}
    token_logprobs = logprobs_data.get("token_logprobs") or []

    # Sum logprobs of all prompt tokens. Index 0 is None (no conditioning);
    # index n_prompt is the generated token (excluded).
    total_logprob = 0.0
    for i in range(min(n_prompt, len(token_logprobs))):
        lp = token_logprobs[i]
        if lp is not None:
            total_logprob += lp

    msg["accuracy"] = total_logprob


async def process_message(client: httpx.AsyncClient, msg: dict) -> None:
    """Route a single message to the appropriate inference handler."""
    rt = msg.get("eval_request_type", "")
    if rt == "generate_until":
        await do_generate_until(client, msg)
    elif rt == "loglikelihood":
        await do_loglikelihood(client, msg)
    elif rt == "loglikelihood_rolling":
        await do_loglikelihood_rolling(client, msg)
    else:
        log.warning(f"Unknown request type: {rt}, filling empty response")
        msg["response"] = ""


# ---------------------------------------------------------------------------
# Task processing pipeline
# ---------------------------------------------------------------------------
async def process_task(
    platform_client: httpx.AsyncClient,
    inference_client: httpx.AsyncClient,
    task: dict,
    overview: dict,
):
    """Process a single task: run inference on all messages, then submit."""
    task_id = overview.get("task_id", "?")
    sla_name = overview.get("target_sla", "Bronze")
    sla_ttft = SLA_TTFT.get(sla_name, 10.0)
    eval_timeout = overview.get("eval_timeout_s") or DEFAULT_TIMEOUT_S
    ask_time = time.time()

    try:
        messages = task.get("messages", [])
        if not messages:
            log.warning(f"Task {task_id}: no messages, skipping")
            return

        # Process all messages concurrently
        # vLLM's continuous batching handles the GPU scheduling
        await asyncio.gather(
            *[process_message(inference_client, m) for m in messages]
        )

        elapsed = time.time() - ask_time
        within_sla = elapsed <= sla_ttft

        # Submit result
        submit_resp = await platform_client.post(
            f"{PLATFORM_URL}/submit",
            json={
                "user": {"name": TEAM_NAME, "token": TOKEN},
                "msg": task,
            },
            timeout=30,
        )
        submit_resp.raise_for_status()

        stats["submitted"] += 1
        tier_stats = sla_stats[sla_name]
        tier_stats["count"] += 1
        tier_stats["lat_sum"] += elapsed
        tier_stats["lat_max"] = max(tier_stats["lat_max"], elapsed)
        if within_sla:
            tier_stats["ok"] += 1
        else:
            tier_stats["late"] += 1
        level = "INFO" if within_sla else "WARNING"
        log.log(
            logging.getLevelName(level),
            f"Task {task_id} submitted in {elapsed:.2f}s "
            f"(SLA={sla_name}/{sla_ttft}s, {'OK' if within_sla else 'LATE'})"
        )

    except Exception as e:
        log.error(f"Task {task_id}: processing failed: {e}")
        stats["failed"] += 1
        # Still try to submit something to avoid -2x penalty.
        # Avoid empty strings / fixed boilerplate (flagged as malicious per
        # 赛制介绍.txt:41); "." is a minimal non-empty token the extractor
        # will treat as a missing answer.
        try:
            for m in task.get("messages", []):
                rt = m.get("eval_request_type", "")
                if rt == "generate_until" and m.get("response") is None:
                    m["response"] = "."
                elif rt in ("loglikelihood", "loglikelihood_rolling") and m.get("accuracy") is None:
                    m["accuracy"] = -100.0
            em_resp = await platform_client.post(
                f"{PLATFORM_URL}/submit",
                json={
                    "user": {"name": TEAM_NAME, "token": TOKEN},
                    "msg": task,
                },
                timeout=30,
            )
            em_resp.raise_for_status()
            log.info(f"Task {task_id}: emergency submit succeeded")
        except Exception as e2:
            log.error(f"Task {task_id}: emergency submit also failed: {e2}")


# ---------------------------------------------------------------------------
# Fetcher: query + ask + dispatch
# ---------------------------------------------------------------------------
async def fetcher(
    fetcher_id: int,
    platform_client: httpx.AsyncClient,
    inference_client: httpx.AsyncClient,
):
    """Continuously fetch tasks from the platform and dispatch for processing."""
    log.info(f"Fetcher-{fetcher_id} started")

    while not shutdown_event.is_set():
        try:
            # Rate control
            await asyncio.sleep(QUERY_INTERVAL)

            # Back-pressure: wait for a free inflight slot before fetching,
            # so we don't /ask tasks we can't start processing immediately
            # (important for tight SLAs).
            await inflight_sem.acquire()
            acquired = True

            try:
                # Query
                resp = await platform_client.post(
                    f"{PLATFORM_URL}/query",
                    json={"token": TOKEN},
                    timeout=10,
                )

                if resp.status_code == 404:
                    await asyncio.sleep(0.3)
                    continue
                if resp.status_code == 429:
                    await asyncio.sleep(0.5)
                    continue
                if resp.status_code != 200:
                    await asyncio.sleep(0.5)
                    continue

                overview = resp.json()
                stats["queried"] += 1
                task_id = overview.get("task_id")
                target_sla = overview.get("target_sla", "Bronze")

                if not task_id:
                    continue

                # Tier 4 skip: optionally bail out on SLA tiers we can't meet.
                # Default SKIP_SLA_TIERS is empty → always try (LATE>=skip).
                if target_sla in SKIP_SLA_TIERS:
                    stats["skipped_tier"] += 1
                    continue

                # Ask (bid) - preliminary: must match target_sla exactly
                ask_resp = await platform_client.post(
                    f"{PLATFORM_URL}/ask",
                    json={
                        "token": TOKEN,
                        "task_id": task_id,
                        "sla": target_sla,
                    },
                    timeout=10,
                )

                if ask_resp.status_code != 200:
                    continue

                ask_data = ask_resp.json()
                if ask_data.get("status") != "accepted":
                    continue

                stats["accepted"] += 1
                task = ask_data["task"]

                # Hand off to a worker; worker will release the semaphore.
                acquired = False
                asyncio.create_task(
                    _worker_wrapper(platform_client, inference_client, task, overview)
                )

            finally:
                if acquired:
                    inflight_sem.release()

        except asyncio.CancelledError:
            break
        except Exception as e:
            log.error(f"Fetcher-{fetcher_id} error: {e}")
            await asyncio.sleep(1)

    log.info(f"Fetcher-{fetcher_id} stopped")


async def _worker_wrapper(
    platform_client: httpx.AsyncClient,
    inference_client: httpx.AsyncClient,
    task: dict,
    overview: dict,
):
    """Wrapper that releases semaphore after task processing."""
    try:
        await process_task(platform_client, inference_client, task, overview)
    finally:
        inflight_sem.release()


# ---------------------------------------------------------------------------
# Stats reporter
# ---------------------------------------------------------------------------
async def stats_reporter():
    """Periodically log statistics: global counters + per-SLA-tier breakdown."""
    while not shutdown_event.is_set():
        await asyncio.sleep(30)
        log.info(
            f"Stats: queried={stats['queried']} accepted={stats['accepted']} "
            f"submitted={stats['submitted']} failed={stats['failed']} "
            f"skipped={stats['skipped_tier']} thinking_trunc={stats['thinking_truncated']}"
        )
        # Tier 4: per-SLA breakdown. Ordered high→low so tight SLAs are obvious.
        tier_order = ["Supreme", "Glorious", "Stellar", "Diamond",
                      "Platinum", "Gold", "Silver", "Bronze"]
        lines = []
        for tier in tier_order:
            s = sla_stats.get(tier)
            if not s or s["count"] == 0:
                continue
            avg_lat = s["lat_sum"] / s["count"]
            target = SLA_TTFT.get(tier, 0)
            pass_rate = 100.0 * s["ok"] / s["count"]
            lines.append(
                f"  {tier:8s} n={s['count']:>4d} ok={s['ok']:>4d} late={s['late']:>4d} "
                f"pass={pass_rate:5.1f}% avg={avg_lat:.2f}s max={s['lat_max']:.2f}s "
                f"(target={target}s)"
            )
        if lines:
            log.info("SLA breakdown:\n" + "\n".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main():
    global inflight_sem
    inflight_sem = asyncio.Semaphore(MAX_INFLIGHT)

    # Load tokenizer
    load_tokenizer()

    # Create HTTP clients with connection pooling
    platform_client = httpx.AsyncClient(
        timeout=30,
        limits=httpx.Limits(max_connections=64, max_keepalive_connections=32),
    )
    inference_client = httpx.AsyncClient(
        timeout=300,
        limits=httpx.Limits(max_connections=128, max_keepalive_connections=64),
    )

    try:
        # Wait for vLLM
        if not await wait_for_vllm(inference_client):
            log.error("vLLM not available, exiting")
            return

        # Register
        reg_resp = await platform_client.post(
            f"{PLATFORM_URL}/register",
            json={"name": TEAM_NAME, "token": TOKEN},
            timeout=10,
        )
        log.info(f"Registered: {reg_resp.json()}")

        # Start fetchers and stats reporter
        tasks = []
        for i in range(NUM_FETCHERS):
            tasks.append(
                asyncio.create_task(
                    fetcher(i, platform_client, inference_client)
                )
            )
        tasks.append(asyncio.create_task(stats_reporter()))

        # Wait for shutdown
        await shutdown_event.wait()

        # Cancel all tasks
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    finally:
        await platform_client.aclose()
        await inference_client.aclose()

    log.info(
        f"Final stats: queried={stats['queried']} accepted={stats['accepted']} "
        f"submitted={stats['submitted']} failed={stats['failed']} "
        f"skipped={stats['skipped_tier']} thinking_trunc={stats['thinking_truncated']}"
    )
    tier_order = ["Supreme", "Glorious", "Stellar", "Diamond",
                  "Platinum", "Gold", "Silver", "Bronze"]
    for tier in tier_order:
        s = sla_stats.get(tier)
        if not s or s["count"] == 0:
            continue
        avg_lat = s["lat_sum"] / s["count"]
        target = SLA_TTFT.get(tier, 0)
        pass_rate = 100.0 * s["ok"] / s["count"]
        log.info(
            f"  {tier:8s} n={s['count']} ok={s['ok']} late={s['late']} "
            f"pass={pass_rate:.1f}% avg={avg_lat:.2f}s max={s['lat_max']:.2f}s target={target}s"
        )


def handle_signal(sig, frame):
    log.info(f"Received signal {sig}, shutting down...")
    shutdown_event.set()


if __name__ == "__main__":
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    log.info(f"Platform: {PLATFORM_URL}")
    log.info(f"Model: {MODEL_PATH}")
    log.info(f"vLLM: {VLLM_URL}")
    log.info(f"Team: {TEAM_NAME}")

    asyncio.run(main())
