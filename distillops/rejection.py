"""Adaptive rejection sampling: ask the teacher until its answer is verified.

For each example we sample up to `max_attempts` times and stop at the first
accepted answer. Compared with a fixed k samples this spends API money only
where it is needed, and the attempt count doubles as a difficulty signal.

Outputs (all JSONL, append-only so a run can be resumed):
- <out>/accepted.jsonl : SFT rows (task.sft_example) + metadata
- <out>/log.jsonl      : one line per example with every attempt's verdict
- <out>/stats.json     : yield, attempts histogram, rejection reasons, cost
"""

from __future__ import annotations

import asyncio
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Sequence

from tqdm.asyncio import tqdm_asyncio

from .io import append_jsonl, read_jsonl, write_json
from .task import Task
from .teacher import Teacher


class _Stopped(Exception):
    """Raised when `stop_when()` turns true (e.g. peak pricing began)."""


async def _one(
    task: Task, teacher: Teacher, ex: Any, max_attempts: int, stop_when: Callable[[], bool] | None = None
) -> dict[str, Any]:
    attempts = []
    for attempt in range(max_attempts):
        if stop_when is not None and stop_when():
            raise _Stopped  # example is not logged, so a later run redoes it (paid calls replay from cache)
        messages = task.teacher_messages(ex, attempt)
        try:
            comp = await teacher.complete(messages, sample_id=attempt)
        except Exception as e:  # keep going; one bad request must not kill the run
            attempts.append({"attempt": attempt, "reason": f"api_error:{type(e).__name__}"})
            continue
        parsed = task.parse_teacher(ex, comp.text)
        verdict = task.verify(ex, parsed, comp.text, attempt)
        attempts.append(
            {
                "attempt": attempt,
                "reason": verdict.reason,
                "info": verdict.info,
                "completion_tokens": comp.completion_tokens,
            }
        )
        if verdict.accepted:
            row = task.sft_example(ex, parsed, comp.text)
            row["meta"] = {"id": task.example_id(ex), "attempts": attempt + 1, "task": task.name}
            return {"id": task.example_id(ex), "accepted": True, "attempts": attempts, "sft": row}
    return {"id": task.example_id(ex), "accepted": False, "attempts": attempts}


def summarize(results: list[dict[str, Any]], meter_summary: dict[str, Any]) -> dict[str, Any]:
    n = len(results)
    acc = [r for r in results if r["accepted"]]
    reasons = Counter(a["reason"] for r in results for a in r["attempts"] if a["reason"] != "ok")
    hist = Counter(len(r["attempts"]) for r in acc)
    return {
        "examples": n,
        "accepted": len(acc),
        "yield": round(len(acc) / n, 4) if n else 0.0,
        "attempts_until_accept": dict(sorted(hist.items())),
        "rejection_reasons": dict(reasons.most_common()),
        "teacher": meter_summary,
    }


async def rejection_sample(
    task: Task,
    teacher: Teacher,
    examples: Sequence[Any],
    out_dir: str | Path,
    max_attempts: int = 3,
    stop_when: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log_path, acc_path = out / "log.jsonl", out / "accepted.jsonl"

    done = {r["id"]: r for r in read_jsonl(log_path)} if log_path.exists() else {}
    todo = [ex for ex in examples if task.example_id(ex) not in done]

    # Limit examples in flight (not just API calls), so `stop_when` is checked right before
    # each call instead of once at start-up for every queued example.
    in_flight = asyncio.Semaphore(teacher.cfg.max_concurrency)

    async def run(ex: Any) -> dict[str, Any] | None:
        async with in_flight:
            try:
                r = await _one(task, teacher, ex, max_attempts, stop_when)
            except _Stopped:
                return None
        sft = r.pop("sft", None)
        if sft is not None:
            append_jsonl(acc_path, sft)
        append_jsonl(log_path, r)  # written as soon as each example finishes
        return r

    new = await tqdm_asyncio.gather(*(run(ex) for ex in todo), desc=f"rejection[{task.name}]")
    stopped = sum(r is None for r in new)
    results = list(done.values()) + [r for r in new if r is not None]
    stats = summarize(results, teacher.meter.summary())
    stats["not_run_stopped"] = stopped  # > 0 means rerun the same command later to finish
    write_json(out / "stats.json", stats)
    return stats


def run_sync(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return asyncio.run(rejection_sample(*args, **kwargs))
