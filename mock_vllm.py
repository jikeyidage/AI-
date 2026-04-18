"""
Mock vLLM server for local integration testing (no GPU / no real model).

Implements the minimal subset of vLLM's OpenAI-compatible API that
submission/client.py actually uses:
  - GET  /v1/models
  - POST /v1/completions  (with echo/logprobs for loglikelihood, sampling for generate_until)

Accepts `prompt` as either a string or a list of token IDs (integers) —
matches vLLM's behavior, and lets the client pass explicit token IDs to
avoid tokenization boundary issues.

Modes:
  - random (default): fake text + random negative logprobs (pipeline test)
  - deterministic:    each token_logprob = -1.0, generated text = "ok"
    Enables correctness unit tests for the client's logprob math.

Run:
    python mock_vllm.py --port 8000 --model-name Qwen3-32B
    python mock_vllm.py --port 8000 --deterministic   # for correctness tests
"""

import argparse
import asyncio
import random
import time
from typing import List, Optional, Union

from fastapi import FastAPI
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Fake text generator
# ---------------------------------------------------------------------------
_WORDS = [
    "yes", "no", "true", "false", "because", "therefore", "however",
    "the", "model", "answer", "is", "correct", "analysis", "shows",
    "conclusion", "result", "value", "equal", "greater", "less",
    "A", "B", "C", "D", "choice", "option", "question",
]


def _fake_text(max_tokens: int, stop: List[str]) -> str:
    """Generate fake text up to max_tokens words, stopping at any stop string."""
    n = random.randint(1, max(1, max_tokens))
    words = [random.choice(_WORDS) for _ in range(n)]
    text = " ".join(words)
    # Maybe append a stop token so the client exercises its strip logic.
    if stop and random.random() < 0.5:
        text = text + random.choice(stop)
    return text


def _fake_logprobs(n_tokens: int) -> dict:
    """Produce a token_logprobs array of length n_tokens (first entry is None)."""
    lps: List[Optional[float]] = [None]
    for _ in range(n_tokens - 1):
        lps.append(round(random.uniform(-6.0, -0.1), 4))
    tokens = [f"t{i}" for i in range(n_tokens)]
    text_offset = list(range(n_tokens))
    return {
        "tokens": tokens,
        "token_logprobs": lps,
        "text_offset": text_offset,
        "top_logprobs": [None] * n_tokens,
    }


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------
class CompletionRequest(BaseModel):
    model: str
    prompt: Union[str, List[str], List[int]]
    max_tokens: int = 16
    temperature: float = 1.0
    top_p: float = 1.0
    stop: Union[str, List[str], None] = None
    logprobs: Optional[int] = None
    echo: bool = False
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    n: int = 1
    extra_body: Optional[dict] = None


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="mock vLLM")

SERVED_MODEL_NAME = "Qwen3-32B"
LATENCY_MS = 0           # set via CLI
DETERMINISTIC = False    # set via CLI — for correctness unit tests


def _count_prompt_tokens(prompt) -> int:
    """Return the token count, whether prompt is a string or list of ints."""
    if isinstance(prompt, list) and prompt and isinstance(prompt[0], int):
        return len(prompt)
    # string / list-of-strings fallback: ~3 chars/token
    text = prompt if isinstance(prompt, str) else prompt[0]
    return max(1, len(text) // 3)


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": SERVED_MODEL_NAME,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "mock",
            }
        ],
    }


@app.post("/v1/completions")
async def completions(req: CompletionRequest):
    if LATENCY_MS > 0:
        await asyncio.sleep(LATENCY_MS / 1000.0)  # async sleep → allow concurrency

    stop = []
    if isinstance(req.stop, str):
        stop = [req.stop]
    elif isinstance(req.stop, list):
        stop = req.stop

    if req.echo:
        # loglikelihood path — echo the prompt and return one logprob per token.
        n_prompt_tokens = _count_prompt_tokens(req.prompt)
        n_total = n_prompt_tokens + max(0, req.max_tokens)
        if DETERMINISTIC:
            # Every prompt token after the first gets logprob -1.0; generated gets -2.0.
            lps: List[Optional[float]] = [None] + [-1.0] * (n_prompt_tokens - 1)
            lps += [-2.0] * max(0, req.max_tokens)
            logprobs = {
                "tokens": [f"t{i}" for i in range(n_total)],
                "token_logprobs": lps,
                "text_offset": list(range(n_total)),
                "top_logprobs": [None] * n_total,
            }
        else:
            logprobs = _fake_logprobs(n_total) if req.logprobs else None

        echoed_text = req.prompt if isinstance(req.prompt, str) else "<token-ids>"
        gen = " ok" if req.max_tokens > 0 else ""
        return {
            "id": "mock-cmpl",
            "object": "text_completion",
            "created": int(time.time()),
            "model": req.model,
            "choices": [
                {
                    "index": 0,
                    "text": echoed_text + gen,
                    "logprobs": logprobs,
                    "finish_reason": "length",
                }
            ],
            "usage": {"prompt_tokens": n_prompt_tokens, "completion_tokens": req.max_tokens, "total_tokens": n_total},
        }

    # generate_until path
    prompt_str = req.prompt if isinstance(req.prompt, str) else str(req.prompt[:20])
    text = "ok" if DETERMINISTIC else _fake_text(req.max_tokens, stop)
    return {
        "id": "mock-cmpl",
        "object": "text_completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "text": text,
                "logprobs": None,
                "finish_reason": "stop" if stop else "length",
            }
        ],
        "usage": {"prompt_tokens": _count_prompt_tokens(req.prompt), "completion_tokens": len(text.split()), "total_tokens": 0},
    }


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", type=str, default="127.0.0.1")
    ap.add_argument("--model-name", type=str, default="Qwen3-32B")
    ap.add_argument("--latency-ms", type=int, default=0,
                    help="Simulate per-request latency in milliseconds")
    ap.add_argument("--deterministic", action="store_true",
                    help="Return fixed logprobs (-1 per token) for correctness tests")
    args = ap.parse_args()

    SERVED_MODEL_NAME = args.model_name
    LATENCY_MS = args.latency_ms
    DETERMINISTIC = args.deterministic

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
