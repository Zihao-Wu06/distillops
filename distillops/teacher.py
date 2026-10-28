"""Black-box teacher client for any OpenAI-compatible chat-completions endpoint.

Features that matter for a distillation run:
- disk cache: re-running a pipeline never pays twice for the same request;
- `sample_id`: distinct cache keys for repeated samples of one prompt
  (needed by rejection sampling, where we retry the same question);
- bounded concurrency + exponential-backoff retries;
- a cost meter in the currency of the configured prices.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Price:
    """Price per 1M tokens, in any currency (the cost meter reports in the same one).
    `input_cache_hit` applies to prompt tokens the provider served from its prompt cache."""

    input_cache_hit: float = 0.0
    input_cache_miss: float = 0.0
    output: float = 0.0


@dataclass
class TeacherConfig:
    base_url: str
    model: str
    api_key_env: str = "TEACHER_API_KEY"  # name of the environment variable holding the key
    temperature: float = 1.0
    max_tokens: int = 1024
    max_concurrency: int = 32
    timeout_s: float = 300.0
    max_retries: int = 6
    extra_body: dict[str, Any] = field(default_factory=dict)
    price: Price = field(default_factory=Price)
    cache_path: str = "cache/teacher.sqlite"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TeacherConfig":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        raw["price"] = Price(**(raw.get("price") or {}))
        return cls(**raw)


def cached_prompt_tokens(usage: Any) -> int:
    """Prompt tokens the provider served from its prompt cache. Providers report this
    differently: DeepSeek as `prompt_cache_hit_tokens`, OpenAI-style APIs as
    `prompt_tokens_details.cached_tokens`; 0 when neither is present."""
    direct = getattr(usage, "prompt_cache_hit_tokens", None)
    if direct:
        return int(direct)
    details = getattr(usage, "prompt_tokens_details", None)
    if details is None:
        return 0
    cached = details.get("cached_tokens") if isinstance(details, dict) else getattr(details, "cached_tokens", None)
    return int(cached or 0)


@dataclass
class Completion:
    text: str
    reasoning: str | None
    prompt_tokens: int
    completion_tokens: int
    cache_hit_tokens: int
    cached: bool  # served from our local disk cache (cost 0)
    cost: float  # in the currency of the configured prices


class CostMeter:
    def __init__(self) -> None:
        self.calls = 0
        self.cache_served = 0
        self.prompt_tokens = 0
        self.cache_hit_tokens = 0
        self.completion_tokens = 0
        self.cost = 0.0

    def add(self, c: Completion) -> None:
        self.calls += 1
        if c.cached:
            self.cache_served += 1
            return
        self.prompt_tokens += c.prompt_tokens
        self.cache_hit_tokens += c.cache_hit_tokens
        self.completion_tokens += c.completion_tokens
        self.cost += c.cost

    def summary(self) -> dict[str, Any]:
        paid = self.calls - self.cache_served
        return {
            "calls": self.calls,
            "served_from_local_cache": self.cache_served,
            "paid_calls": paid,
            "prompt_tokens": self.prompt_tokens,
            "api_cache_hit_tokens": self.cache_hit_tokens,
            "completion_tokens": self.completion_tokens,
            "avg_completion_tokens": round(self.completion_tokens / paid, 1) if paid else 0,
            "cost": round(self.cost, 4),
        }


class _DiskCache:
    def __init__(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT)")
        self.db.commit()

    def get(self, key: str) -> dict | None:
        row = self.db.execute("SELECT v FROM kv WHERE k=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key: str, value: dict) -> None:
        self.db.execute("INSERT OR REPLACE INTO kv VALUES (?,?)", (key, json.dumps(value, ensure_ascii=False)))
        self.db.commit()


def _cost(cfg: TeacherConfig, prompt: int, hit: int, completion: int) -> float:
    p = cfg.price
    miss = max(prompt - hit, 0)
    return (hit * p.input_cache_hit + miss * p.input_cache_miss + completion * p.output) / 1e6


class Teacher:
    def __init__(self, cfg: TeacherConfig, client: Any | None = None) -> None:
        self.cfg = cfg
        self.meter = CostMeter()
        self._sem = asyncio.Semaphore(cfg.max_concurrency)
        self._cache = _DiskCache(cfg.cache_path)
        if client is None:  # real API client; tests inject a fake one
            from openai import AsyncOpenAI

            key = os.environ.get(cfg.api_key_env)
            if not key:
                raise RuntimeError(f"Set the {cfg.api_key_env} environment variable first.")
            client = AsyncOpenAI(base_url=cfg.base_url, api_key=key, timeout=cfg.timeout_s, max_retries=0)
        self.client = client

    def _key(self, messages: list[dict], sample_id: int, params: dict) -> str:
        blob = json.dumps(
            {"model": self.cfg.model, "messages": messages, "sample_id": sample_id, "params": params},
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    async def complete(self, messages: list[dict], sample_id: int = 0, **overrides: Any) -> Completion:
        params = {
            "temperature": overrides.get("temperature", self.cfg.temperature),
            "max_tokens": overrides.get("max_tokens", self.cfg.max_tokens),
        }
        extra = {**self.cfg.extra_body, **overrides.get("extra_body", {})}
        key = self._key(messages, sample_id, {**params, "extra_body": extra})

        hit = self._cache.get(key)
        if hit is not None:
            c = Completion(**{**hit, "cached": True, "cost": 0.0})
            self.meter.add(c)
            return c

        async with self._sem:
            resp = await self._call_with_retry(messages, params, extra)

        msg = resp.choices[0].message
        usage = resp.usage
        prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion = int(getattr(usage, "completion_tokens", 0) or 0)
        hit_tokens = cached_prompt_tokens(usage)
        c = Completion(
            text=msg.content or "",
            reasoning=getattr(msg, "reasoning_content", None),
            prompt_tokens=prompt,
            completion_tokens=completion,
            cache_hit_tokens=hit_tokens,
            cached=False,
            cost=_cost(self.cfg, prompt, hit_tokens, completion),
        )
        record = {k: v for k, v in c.__dict__.items() if k not in ("cached", "cost")}
        self._cache.put(key, record)
        self.meter.add(c)
        return c

    async def _call_with_retry(self, messages: list[dict], params: dict, extra: dict) -> Any:
        delay = 2.0
        for attempt in range(self.cfg.max_retries + 1):
            try:
                return await self.client.chat.completions.create(
                    model=self.cfg.model, messages=messages, extra_body=extra or None, **params
                )
            except Exception as e:  # rate limits, timeouts, 5xx
                status = getattr(e, "status_code", None)
                if status in (400, 401, 403, 404) or attempt == self.cfg.max_retries:
                    raise
                await asyncio.sleep(delay + random.random())
                delay = min(delay * 2, 60)
        raise RuntimeError("unreachable")
