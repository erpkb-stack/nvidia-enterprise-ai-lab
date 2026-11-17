"""Backend-only client for NVIDIA inference endpoints.

Targets the OpenAI-compatible API exposed by both the hosted NVIDIA API catalog and self-hosted NIM
containers. The API key never leaves the backend and is never logged or echoed in errors.
"""

import json
import logging
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx

from ..config import settings

log = logging.getLogger(__name__)


class UpstreamError(RuntimeError):
    """An NVIDIA endpoint failed. The message is safe to show to users."""


def _headers() -> dict[str, str]:
    h = {"Content-Type": "application/json", "Accept": "application/json"}
    if settings.nvidia_nim_api_key:
        h["Authorization"] = f"Bearer {settings.nvidia_nim_api_key}"
    return h


def _raise_for(resp: httpx.Response, what: str) -> None:
    if resp.status_code < 400:
        return
    detail = ""
    try:
        body = resp.json()
        detail = str(body.get("detail") or body.get("error") or body)[:300]
    except ValueError:
        detail = resp.text[:300]
    log.warning("nvidia %s failed status=%s", what, resp.status_code)
    raise UpstreamError(f"NVIDIA {what} endpoint returned HTTP {resp.status_code}: {detail}")


async def chat(
    messages: list[dict[str, str]], max_tokens: int = 512, temperature: float = 0.2,
    model: str | None = None,
) -> dict[str, Any]:
    payload = {
        "model": model or settings.nvidia_chat_model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    try:
        async with httpx.AsyncClient(timeout=settings.nvidia_timeout_seconds) as client:
            resp = await client.post(
                f"{settings.llm_base_url}/chat/completions", headers=_headers(), json=payload
            )
    except httpx.HTTPError as exc:
        raise UpstreamError(f"Could not reach NVIDIA chat endpoint: {type(exc).__name__}") from exc
    _raise_for(resp, "chat")
    body = resp.json()
    return {
        "text": body["choices"][0]["message"]["content"],
        "usage": body.get("usage", {}),
        "model": body.get("model", payload["model"]),
    }


async def chat_stream_timed(
    client: httpx.AsyncClient, prompt: str, max_tokens: int, model: str | None = None
) -> dict[str, Any]:
    """Run one streaming completion and measure TTFT and inter-token latency on the client side."""
    payload = {
        "model": model or settings.nvidia_chat_model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    start = time.perf_counter()
    first: float | None = None
    chunks = 0
    usage: dict[str, Any] = {}
    async with client.stream(
        "POST", f"{settings.llm_base_url}/chat/completions", headers=_headers(), json=payload
    ) as resp:
        if resp.status_code >= 400:
            await resp.aread()
            _raise_for(resp, "chat")
        async for line in _sse_lines(resp):
            if line == "[DONE]":
                break
            event = json.loads(line)
            if event.get("usage"):
                usage = event["usage"]
            choices = event.get("choices") or []
            if choices and choices[0].get("delta", {}).get("content"):
                chunks += 1
                if first is None:
                    first = time.perf_counter()
    end = time.perf_counter()
    out_tokens = int(usage.get("completion_tokens") or chunks)
    ttft = (first or end) - start
    decode = max(end - (first or end), 0.0)
    return {
        "ttft_s": ttft,
        "e2e_s": end - start,
        "output_tokens": out_tokens,
        "input_tokens": int(usage.get("prompt_tokens") or 0),
        "itl_s": decode / max(out_tokens - 1, 1),
    }


async def _sse_lines(resp: httpx.Response) -> AsyncIterator[str]:
    async for raw in resp.aiter_lines():
        raw = raw.strip()
        if raw.startswith("data:"):
            yield raw[5:].strip()


async def embed(texts: list[str], input_type: str = "passage") -> list[list[float]]:
    vectors: list[list[float]] = []
    async with httpx.AsyncClient(timeout=settings.nvidia_timeout_seconds) as client:
        for i in range(0, len(texts), 32):
            batch = texts[i : i + 32]
            try:
                resp = await client.post(
                    f"{settings.llm_base_url}/embeddings",
                    headers=_headers(),
                    json={
                        "model": settings.nvidia_embed_model,
                        "input": batch,
                        "input_type": input_type,
                        "encoding_format": "float",
                        "truncate": "END",
                    },
                )
            except httpx.HTTPError as exc:
                raise UpstreamError(
                    f"Could not reach NVIDIA embedding endpoint: {type(exc).__name__}"
                ) from exc
            _raise_for(resp, "embedding")
            data = sorted(resp.json()["data"], key=lambda d: d["index"])
            vectors.extend(d["embedding"] for d in data)
    return vectors


async def rerank(query: str, passages: list[str]) -> list[float]:
    """Return one relevance logit per passage, in the original passage order."""
    try:
        async with httpx.AsyncClient(timeout=settings.nvidia_timeout_seconds) as client:
            resp = await client.post(
                settings.nvidia_rerank_url,
                headers=_headers(),
                json={
                    "model": settings.nvidia_rerank_model,
                    "query": {"text": query},
                    "passages": [{"text": p} for p in passages],
                    "truncate": "END",
                },
            )
    except httpx.HTTPError as exc:
        raise UpstreamError(f"Could not reach NVIDIA rerank endpoint: {type(exc).__name__}") from exc
    _raise_for(resp, "rerank")
    scores = [0.0] * len(passages)
    for r in resp.json().get("rankings", []):
        scores[r["index"]] = float(r["logit"])
    return scores


async def probe() -> dict[str, Any]:
    """Cheap connectivity check used by the readiness endpoint and the UI."""
    if not settings.llm_enabled:
        return {"configured": False, "reachable": False, "mode": "fallback",
                "message": "No NVIDIA API key or self-hosted NIM URL configured."}
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{settings.llm_base_url}/models", headers=_headers())
        ok = resp.status_code < 400
        return {"configured": True, "reachable": ok, "mode": "live" if ok else "error",
                "base_url": settings.llm_base_url, "chat_model": settings.nvidia_chat_model,
                "message": "Endpoint reachable." if ok else f"HTTP {resp.status_code} from /models"}
    except httpx.HTTPError as exc:
        return {"configured": True, "reachable": False, "mode": "error",
                "base_url": settings.llm_base_url, "message": type(exc).__name__}
