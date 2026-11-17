"""Live load test against the configured OpenAI-compatible NVIDIA endpoint (hosted or self-hosted NIM).

These are real client-side measurements of whatever endpoint is configured, including network time.
They are not MLPerf results and describe that endpoint only.
"""

import asyncio
import statistics
import time

import httpx

from ..config import settings
from . import nvidia

_WORDS = ("enterprise accelerated computing inference retrieval cluster latency throughput "
          "kubernetes validation architecture deployment").split()


def synthetic_prompt(approx_tokens: int) -> str:
    # ~0.75 words per token for English text
    words = [_WORDS[i % len(_WORDS)] for i in range(max(int(approx_tokens * 0.75), 1))]
    return ("Summarize the following notes in detail and keep writing until stopped:\n"
            + " ".join(words))


def _pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


async def run(requests: int, concurrency: int, input_tokens: int, output_tokens: int,
              model: str | None = None) -> dict:
    prompt = synthetic_prompt(input_tokens)
    sem = asyncio.Semaphore(concurrency)
    results: list[dict] = []
    errors: list[str] = []

    async with httpx.AsyncClient(timeout=settings.nvidia_timeout_seconds) as client:
        async def one() -> None:
            async with sem:
                try:
                    results.append(await nvidia.chat_stream_timed(client, prompt, output_tokens, model))
                except (nvidia.UpstreamError, httpx.HTTPError, ValueError) as exc:
                    errors.append(str(exc)[:200] if isinstance(exc, nvidia.UpstreamError)
                                  else type(exc).__name__)

        start = time.perf_counter()
        await asyncio.gather(*(one() for _ in range(requests)))
        wall = time.perf_counter() - start

    ttft = [r["ttft_s"] * 1000 for r in results]
    itl = [r["itl_s"] * 1000 for r in results]
    e2e = [r["e2e_s"] * 1000 for r in results]
    out_tokens = sum(r["output_tokens"] for r in results)

    def stats(v: list[float]) -> dict:
        return {"mean": round(statistics.fmean(v), 1) if v else 0.0,
                "p50": round(_pct(v, 0.5), 1), "p90": round(_pct(v, 0.9), 1),
                "p99": round(_pct(v, 0.99), 1)}

    return {
        "mode": "LIVE_MEASUREMENT",
        "endpoint": settings.llm_base_url,
        "model": model or settings.nvidia_chat_model,
        "requests": requests, "concurrency": concurrency,
        "succeeded": len(results), "failed": len(errors), "errors": errors[:5],
        "wall_time_s": round(wall, 2),
        "ttft_ms": stats(ttft), "inter_token_latency_ms": stats(itl), "e2e_latency_ms": stats(e2e),
        "output_tokens_per_s": round(out_tokens / wall, 1) if wall else 0.0,
        "requests_per_s": round(len(results) / wall, 2) if wall else 0.0,
        "mean_input_tokens": round(statistics.fmean([r["input_tokens"] for r in results]), 0)
        if results else 0,
        "note": "Client-side measurement including network latency to the endpoint. Not an MLPerf result.",
    }
