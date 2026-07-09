"""Async client for the ai-router LiteLLM gateway (OpenAI-compatible).

Two capabilities are used:
  * prompt scoring — vLLM's `echo=true, max_tokens=0, logprobs=1` returns
    `prompt_logprobs`: for every input token, its logprob and exact rank under
    the model. This is what powers the deterministic perplexity / rank signals.
  * chat — plain chat completion for the LLM-judge signal.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

import httpx

from . import config

_client: httpx.AsyncClient | None = None


def client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(
            timeout=httpx.Timeout(180.0, connect=10.0),
            headers={"Authorization": f"Bearer {config.ROUTER_API_KEY}"},
        )
    return _client


@dataclass
class ScoredToken:
    text: str
    logprob: float
    rank: int


async def score_prompt(text: str, model: str) -> list[ScoredToken]:
    """Per-token logprob+rank of `text` under `model` (no generation happens).

    vLLM's prompt_logprobs entry per position is a dict token_id -> info holding
    the top-k tokens plus the actual token. With logprobs=1 the parse is
    unambiguous: one entry means the actual token was the model's top-1 pick;
    two entries mean the actual token (rank > 1) rode along with the top-1.
    """
    resp = await client().post(
        f"{config.ROUTER_BASE_URL}/completions",
        json={
            "model": model,
            "prompt": text,
            "max_tokens": 0,
            "echo": True,
            "logprobs": 1,
            "temperature": 0,
        },
    )
    resp.raise_for_status()
    data = resp.json()
    entries = data["choices"][0].get("prompt_logprobs") or []
    tokens: list[ScoredToken] = []
    for entry in entries:
        if not entry:  # first token has no context -> null
            continue
        infos = list(entry.values())
        if len(infos) == 1:
            actual = infos[0]
        else:
            non_top = [i for i in infos if i.get("rank", 1) != 1]
            actual = non_top[0] if non_top else infos[0]
        tokens.append(
            ScoredToken(
                text=actual.get("decoded_token", ""),
                logprob=float(actual["logprob"]),
                rank=int(actual.get("rank", 1)),
            )
        )
    return tokens


async def chat(
    messages: list[dict],
    model: str,
    max_tokens: int = 700,
    temperature: float = 0.0,
) -> str:
    resp = await client().post(
        f"{config.ROUTER_BASE_URL}/chat/completions",
        json={
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"] or ""


def extract_json(raw: str) -> dict | None:
    """Parse the first JSON object out of a model reply (tolerates fences/prose)."""
    raw = raw.strip()
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
