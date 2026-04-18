"""
Correctness tests for do_loglikelihood and do_loglikelihood_rolling.

Requires a deterministic mock_vllm running on port 8000:
    python mock_vllm.py --port 8000 --deterministic

Mock returns: token_logprobs = [None, -1, -1, ..., -1 (prompt echo), -2, -2, ...]
            where prompt echo has n_prompt entries (first is None), each
            -1.0; generated tokens are -2.0.

Expected client behavior with FallbackTokenizer (1 token per 3 chars):
  - do_loglikelihood: sums continuation tokens → -(n_cont)
  - do_loglikelihood_rolling: sums prompt tokens (skip first None) → -(n_prompt - 1)

Exit code 0 iff all cases pass.
"""

import asyncio
import os
import sys

# Point client at mock_vllm; force fallback tokenizer by pointing at a
# nonexistent model path.
os.environ["MODEL_PATH"] = "__nonexistent__"
os.environ["VLLM_PORT"] = "8000"
os.environ["PLATFORM_URL"] = "http://127.0.0.1:9999"  # not used by tests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "submission"))

import httpx

import client as C


def _expected_ll(prompt: str, cont: str) -> float:
    n_cont = len(C.tokenizer.encode(cont, add_special_tokens=False))
    return -1.0 * n_cont


def _expected_ll_rolling(prompt: str) -> float:
    n_prompt = len(C.tokenizer.encode(prompt, add_special_tokens=False))
    # index 0 is None → skipped, so we sum n_prompt-1 entries of -1.0
    return -1.0 * max(0, n_prompt - 1)


async def run():
    C.load_tokenizer()
    assert isinstance(C.tokenizer, C._FallbackTokenizer), (
        f"Test expects FallbackTokenizer, got {type(C.tokenizer).__name__}. "
        f"Set MODEL_PATH to a nonexistent path to force fallback."
    )

    # Probe mock is up
    async with httpx.AsyncClient(timeout=10) as probe:
        r = await probe.get(f"{C.VLLM_URL}/v1/models")
        assert r.status_code == 200, f"mock_vllm not reachable: {r.status_code}"

    cases_ll = [
        ("hello world this is a prompt", " yes"),
        ("ABC", " correct"),
        ("A" * 100, " B" * 5),
        ("short", " longer continuation here"),
    ]
    cases_lr = [
        "hello world this is a prompt",
        "short",
        "A" * 300,
    ]

    client = httpx.AsyncClient(timeout=30)
    fails = 0

    print("=== do_loglikelihood ===")
    for prompt, cont in cases_ll:
        msg = {
            "prompt": prompt,
            "eval_continuation": cont,
            "eval_request_type": "loglikelihood",
        }
        await C.do_loglikelihood(client, msg)
        expected = _expected_ll(prompt, cont)
        got = msg["accuracy"]
        ok = abs(got - expected) < 1e-6
        fails += 0 if ok else 1
        status = "OK" if ok else "FAIL"
        print(f"  [{status}] prompt_len={len(prompt):4d} cont_len={len(cont):3d} "
              f"expected={expected:+.3f} got={got:+.3f}")

    print("=== do_loglikelihood_rolling ===")
    for prompt in cases_lr:
        msg = {
            "prompt": prompt,
            "eval_request_type": "loglikelihood_rolling",
        }
        await C.do_loglikelihood_rolling(client, msg)
        expected = _expected_ll_rolling(prompt)
        got = msg["accuracy"]
        ok = abs(got - expected) < 1e-6
        fails += 0 if ok else 1
        status = "OK" if ok else "FAIL"
        print(f"  [{status}] prompt_len={len(prompt):4d} "
              f"expected={expected:+.3f} got={got:+.3f}")

    # Empty-continuation edge case
    print("=== edge cases ===")
    msg_empty_cont = {"prompt": "hello", "eval_continuation": ""}
    await C.do_loglikelihood(client, msg_empty_cont)
    ok_ec = msg_empty_cont["accuracy"] == 0.0
    print(f"  [{'OK' if ok_ec else 'FAIL'}] empty continuation → {msg_empty_cont['accuracy']}")
    fails += 0 if ok_ec else 1

    msg_empty_prompt = {"prompt": ""}
    await C.do_loglikelihood_rolling(client, msg_empty_prompt)
    ok_ep = msg_empty_prompt["accuracy"] == 0.0
    print(f"  [{'OK' if ok_ep else 'FAIL'}] empty prompt (rolling) → {msg_empty_prompt['accuracy']}")
    fails += 0 if ok_ep else 1

    await client.aclose()
    print(f"\n{'PASS' if fails == 0 else 'FAIL'} ({fails} failures)")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(run()))
