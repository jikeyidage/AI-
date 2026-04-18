"""
Warm up vLLM before real traffic hits.

Why this exists:
  vLLM captures CUDA graphs lazily. The first request at a given (batch, seqlen)
  shape triggers a graph capture that can take 2-10 seconds. If that cost lands
  on the first real task's SLA clock, we're already late. Sending a handful of
  dummy requests now amortizes that cost outside the scored window.

We cover the shapes the client actually uses:
  - short generate_until (small prompt, small max_tokens)
  - longer generate_until (medium prompt, medium max_tokens)
  - echo+logprobs (the loglikelihood path)
  - a batch of concurrent requests (so the scheduler captures multi-seq graphs)

Exit code is informational — run.sh treats failure as non-fatal.
"""

import argparse
import asyncio
import sys
import time

import httpx


async def _one(client: httpx.AsyncClient, url: str, payload: dict, label: str):
    t0 = time.time()
    try:
        r = await client.post(url, json=payload, timeout=120)
        r.raise_for_status()
        dt = time.time() - t0
        print(f"[warmup] {label}: {dt:.2f}s OK")
    except Exception as e:
        dt = time.time() - t0
        print(f"[warmup] {label}: {dt:.2f}s FAIL ({e})")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()

    url = f"http://{args.host}:{args.port}/v1/completions"
    short_prompt = "Hello, how are you?"
    medium_prompt = "The quick brown fox jumps over the lazy dog. " * 50  # ~2k chars
    long_prompt = "The quick brown fox jumps over the lazy dog. " * 200   # ~9k chars

    gen_short = {"model": args.model, "prompt": short_prompt, "max_tokens": 16, "temperature": 0}
    gen_medium = {"model": args.model, "prompt": medium_prompt, "max_tokens": 64, "temperature": 0}
    echo_req = {"model": args.model, "prompt": short_prompt, "max_tokens": 1,
                "temperature": 0, "logprobs": 1, "echo": True}
    echo_long = {"model": args.model, "prompt": long_prompt[:3500], "max_tokens": 1,
                 "temperature": 0, "logprobs": 1, "echo": True}

    async with httpx.AsyncClient() as client:
        # Serial: exercise single-stream CUDA graphs.
        await _one(client, url, gen_short, "serial/gen_short")
        await _one(client, url, gen_medium, "serial/gen_medium")
        await _one(client, url, echo_req, "serial/echo_short")
        await _one(client, url, echo_long, "serial/echo_long")
        # Concurrent batch: exercise multi-seq scheduling.
        await asyncio.gather(*[
            _one(client, url, gen_short, f"concurrent/{i}") for i in range(8)
        ])
    print("[warmup] done")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
