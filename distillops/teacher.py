"""Black-box teacher client for any OpenAI-compatible chat-completions endpoint.

Features that matter for a distillation run:
- disk cache: re-running a pipeline never sends the same request twice;
- `sample_id`: distinct cache keys for repeated samples of one prompt
  (needed by rejection sampling, where we retry the same question);
- bounded concurrency; retries with exponential backoff for network errors, timeouts, rate
  limits and server errors only (anything else is raised at once, see `is_transient`);
- a usage counter (calls, prompt / cached / completion tokens).
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
class TeacherConfig:
    base_url: str
    model: str
    api_key_env: str = "TEACHER_API_KEY"  # name of the environment variable holding the key
    temperature: float | None = 1.0  # None = not sent (models that only accept their default)
    max_tokens: int | None = 1024  # None = not sent (the provider's default limit)
    # the request field for max_tokens: OpenAI's reasoning models take "max_completion_tokens"
    max_tokens_param: str = "max_tokens"
    max_concurrency: int = 32
    timeout_s: float = 300.0
    max_retries: int = 6
    extra_body: dict[str, Any] = field(default_factory=dict)
    cache_path: str = "cache/teacher.sqlite"

    def __post_init__(self) -> None:
        if self.max_tokens_param not in TOKEN_LIMIT_PARAMS:
            raise ValueError(f"max_tokens_param must be one of {TOKEN_LIMIT_PARAMS} (got {self.max_tokens_param!r}); "
                             "other provider-specific fields go in extra_body")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TeacherConfig":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if "price" in raw:  # configs written for older versions
            raw.pop("price")
            print(f"note: {path}: `price` is no longer used (distillops reports token usage, not money); ignored",
                  flush=True)
        return cls(**raw)

    def request_params(self, **overrides: Any) -> dict[str, Any]:
        """Sampling fields sent with every request; fields set to None are left out."""
        temperature = overrides.get("temperature", self.temperature)
        max_tokens = overrides.get("max_tokens", self.max_tokens)
        params: dict[str, Any] = {}
        if temperature is not None:
            params["temperature"] = temperature
        if max_tokens is not None:
            params[self.max_tokens_param] = max_tokens
        return params

    def identity(self) -> dict[str, Any]:
        """What determines the teacher's answers (used to keep separate runs apart)."""
        return {"base_url": self.base_url, "model": self.model, "params": self.request_params(),
                "extra_body": self.extra_body}


TOKEN_LIMIT_PARAMS = ("max_tokens", "max_completion_tokens")


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
    cached: bool  # served from our local disk cache (no request sent)


class UsageMeter:
    """Calls and tokens of the requests actually sent (local-cache hits are counted apart)."""

    def __init__(self) -> None:
        self.calls = 0
        self.cache_served = 0
        self.prompt_tokens = 0
        self.cache_hit_tokens = 0
        self.completion_tokens = 0

    def add(self, c: Completion) -> None:
        self.calls += 1
        if c.cached:
            self.cache_served += 1
            return
        self.prompt_tokens += c.prompt_tokens
        self.cache_hit_tokens += c.cache_hit_tokens
        self.completion_tokens += c.completion_tokens

    def summary(self) -> dict[str, Any]:
        sent = self.calls - self.cache_served
        return {
            "calls": self.calls,
            "served_from_local_cache": self.cache_served,
            "sent_to_api": sent,
            "prompt_tokens": self.prompt_tokens,
            "api_cache_hit_tokens": self.cache_hit_tokens,
            "completion_tokens": self.completion_tokens,
            "avg_completion_tokens": round(self.completion_tokens / sent, 1) if sent else 0,
        }


RETRY_STATUS = (408, 409, 429)  # plus every 5xx


class BadResponseError(RuntimeError):
    """The endpoint answered, but not with a chat completion (e.g. HTTP 200 with an error body
    from a gateway, or a body that is not valid JSON). Treated like a network error."""


def http_status(e: BaseException) -> int | None:
    status = getattr(e, "status_code", None)
    return status if isinstance(status, int) else None


def is_connection_error(e: BaseException) -> bool:
    """The request did not get an HTTP answer (connection refused or reset, DNS, timeout)."""
    try:
        import openai

        if isinstance(e, openai.APIConnectionError):  # includes APITimeoutError
            return True
    except ImportError:
        pass
    return isinstance(e, (TimeoutError, asyncio.TimeoutError, ConnectionError))


def is_transient(e: BaseException) -> bool:
    """Worth retrying: network errors, timeouts, rate limits and server errors. Everything else
    (a bad request, a wrong key, a missing model, or a bug or broken package on this machine)
    fails the same way on every retry."""
    if isinstance(e, BadResponseError):
        return True
    status = http_status(e)
    if status is not None:
        return status in RETRY_STATUS or status >= 500
    return is_connection_error(e)


def is_request_error(e: BaseException) -> bool:
    """An error about this one request (other requests can still succeed): an HTTP error other
    than a wrong key, missing permission or unknown model, or a network error that outlasted the
    retries. Anything else is a problem with the setup and should stop the run."""
    if isinstance(e, BadResponseError):
        return True
    status = http_status(e)
    if status is not None:
        return status not in (401, 403, 404)
    return is_connection_error(e)


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


class Teacher:
    def __init__(self, cfg: TeacherConfig, client: Any | None = None) -> None:
        self.cfg = cfg
        self.meter = UsageMeter()
        self._sem = asyncio.Semaphore(cfg.max_concurrency)
        self._cache = _DiskCache(cfg.cache_path)
        if client is None:  # real API client; tests inject a fake one
            from openai import AsyncOpenAI

            key = os.environ.get(cfg.api_key_env)
            if not key:
                raise RuntimeError(f"Set the {cfg.api_key_env} environment variable first.")
            client = AsyncOpenAI(base_url=cfg.base_url, api_key=key, timeout=cfg.timeout_s, max_retries=0)
        self.client = client

    def _key(self, messages: list[dict], sample_id: int, params: dict, with_endpoint: bool = True) -> str:
        """Cache key. It includes the endpoint, so two providers serving the same model name never
        share answers; `with_endpoint=False` gives the key of caches written before that."""
        fields = {"model": self.cfg.model, "messages": messages, "sample_id": sample_id, "params": params}
        if with_endpoint:
            fields["base_url"] = self.cfg.base_url
        blob = json.dumps(fields, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _cached(self, messages: list[dict], sample_id: int, params: dict) -> tuple[str, dict | None]:
        key = self._key(messages, sample_id, params)
        hit = self._cache.get(key)
        if hit is None:  # an entry from before the endpoint was recorded (it has no "base_url")
            legacy = self._cache.get(self._key(messages, sample_id, params, with_endpoint=False))
            hit = legacy if legacy is not None and "base_url" not in legacy else None
        return key, hit

    async def complete(self, messages: list[dict], sample_id: int = 0, **overrides: Any) -> Completion:
        params = self.cfg.request_params(**overrides)
        extra = {**self.cfg.extra_body, **overrides.get("extra_body", {})}
        key, hit = self._cached(messages, sample_id, {**params, "extra_body": extra})
        if hit is not None:
            fields = {k: v for k, v in hit.items() if k in Completion.__dataclass_fields__}
            c = Completion(**{**fields, "cached": True})
            self.meter.add(c)
            return c

        async with self._sem:
            resp = await self._call_with_retry(messages, params, extra)

        msg = resp.choices[0].message  # shape checked in _call_with_retry
        usage = getattr(resp, "usage", None)
        prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion = int(getattr(usage, "completion_tokens", 0) or 0)
        hit_tokens = cached_prompt_tokens(usage)
        c = Completion(
            text=msg.content or "",
            reasoning=getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None),
            prompt_tokens=prompt,
            completion_tokens=completion,
            cache_hit_tokens=hit_tokens,
            cached=False,
        )
        record = {k: v for k, v in c.__dict__.items() if k != "cached"}
        record.update(model=self.cfg.model, base_url=self.cfg.base_url)  # lets `usage` count per teacher
        self._cache.put(key, record)
        self.meter.add(c)
        return c

    async def _call_with_retry(self, messages: list[dict], params: dict, extra: dict) -> Any:
        delay = 2.0
        for attempt in range(self.cfg.max_retries + 1):
            try:
                try:
                    resp = await self.client.chat.completions.create(
                        model=self.cfg.model, messages=messages, extra_body=extra or None, **params
                    )
                except json.JSONDecodeError as e:  # a 200 whose body is not JSON
                    raise BadResponseError(f"the endpoint's reply is not valid JSON: {e}") from e
                choices = getattr(resp, "choices", None)
                if not isinstance(choices, list) or not choices or getattr(choices[0], "message", None) is None:
                    error = getattr(resp, "error", None)  # e.g. {"error": {...}} sent with HTTP 200
                    raise BadResponseError(f"the endpoint answered without a completion"
                                           f"{f': {error}' if error else f' ({type(resp).__name__})'}")
                return resp
            except Exception as e:
                if not is_transient(e) or attempt == self.cfg.max_retries:
                    raise
                await asyncio.sleep(delay + random.random())
                delay = min(delay * 2, 60)
        raise RuntimeError("unreachable")
