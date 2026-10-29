"""Adaptive rejection sampling: ask the teacher until its answer is verified.

For each example we sample up to `max_attempts` times and stop at the first
accepted answer. Compared with a fixed k samples this spends API money only
where it is needed, and the attempt count doubles as a difficulty signal.

Outputs (all JSONL, append-only so a run can be resumed):
- <out>/accepted.jsonl : SFT rows (task.sft_example) + metadata
- <out>/log.jsonl      : one line per example with every attempt's verdict
- <out>/stats.json     : yield, attempts histogram, rejection reasons, token usage

An error about one request (an HTTP error, or a network error that outlasted the retries) is
recorded and the run goes on. Any other error (a wrong key or model, or a bug or broken package
on this machine) would hit every request, so it stops the run; finished examples are kept.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Sequence

from tqdm.asyncio import tqdm_asyncio

from .io import append_jsonl, read_jsonl, write_json
from .task import Task, check_ids
from .teacher import Teacher, http_status, is_request_error


class TeacherSetupError(RuntimeError):
    """A teacher call failed in a way every other call would too, so the run stops."""


class _Stopped(Exception):
    """Raised when `stop_when()` turns true (e.g. a deadline or a limit set by the caller was reached)."""


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
        except Exception as e:
            if not is_request_error(e):
                status = http_status(e)
                hint = ("check the API key (api_key_env), base_url and model in the teacher config"
                        if status is not None else
                        "this is not a network or rate-limit error: check the installed packages and the teacher "
                        "config")
                raise TeacherSetupError(f"teacher call failed with {type(e).__name__}"
                                        f"{f' (HTTP {status})' if status else ''}: {e}. Every call would fail the "
                                        f"same way, so the run stops ({hint}). Finished examples are kept; rerun "
                                        "the same command to continue.") from e
            attempts.append({"attempt": attempt, "reason": f"api_error:{type(e).__name__}"})  # this request only
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

    ids = check_ids(task, examples)
    # ids are text; logs written by tasks that returned other types still match
    done = {str(r["id"]): r for r in read_jsonl(log_path)} if log_path.exists() else {}
    todo = [ex for ex, eid in zip(examples, ids) if eid not in done]

    # Limit examples in flight (not just API calls), so `stop_when` is checked right before
    # each call instead of once at start-up for every queued example.
    in_flight = asyncio.Semaphore(teacher.cfg.max_concurrency)
    halted: list[bool] = []  # set by a setup error: no example sends another request

    def should_stop() -> bool:
        return bool(halted) or (stop_when is not None and stop_when())

    async def run(ex: Any) -> dict[str, Any] | None:
        async with in_flight:
            try:
                r = await _one(task, teacher, ex, max_attempts, should_stop)
            except _Stopped:
                return None
            except TeacherSetupError:
                halted.append(True)  # before this slot is released, so a waiting example sees it
                raise
        sft = r.pop("sft", None)
        if sft is not None:
            append_jsonl(acc_path, sft)
        append_jsonl(log_path, r)  # written as soon as each example finishes
        return r

    tasks = [asyncio.ensure_future(run(ex)) for ex in todo]
    try:
        new = await tqdm_asyncio.gather(*tasks, desc=f"rejection[{task.name}]")
    except BaseException:  # e.g. TeacherSetupError: stop every other example too, not just this one
        for t in tasks:
            t.cancel()  # an example is logged whole or not at all: it is cancelled only while awaiting a call
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    stopped = sum(r is None for r in new)
    results = list(done.values()) + [r for r in new if r is not None]
    stats = summarize(results, teacher.meter.summary())
    stats["not_run_stopped"] = stopped  # > 0 means rerun the same command later to finish
    write_json(out / "stats.json", stats)
    return stats


def check_run_settings(out_dir: str | Path, settings: dict[str, Any]) -> None:
    """Record what produced `out_dir` (run.json) and refuse to resume it with different settings:
    examples already in its log are skipped, so another teacher or task setting would silently
    mix two runs into one dataset."""
    import json

    out = Path(out_dir)
    path = out / "run.json"
    settings = json.loads(json.dumps(settings, sort_keys=True, default=str))
    log = out / "log.jsonl"
    has_results = log.exists() and log.stat().st_size > 0
    if path.exists() and has_results:  # a run that produced nothing (e.g. a config typo) may be redone
        before = json.loads(path.read_text(encoding="utf-8"))
        changed = sorted(k for k in set(before) & set(settings) if before[k] != settings[k])
        if changed:
            raise SystemExit(
                f"{out} holds a run with different settings ({', '.join(changed)}); resuming would mix the two.\n"
                f"  before: { {k: before.get(k) for k in changed} }\n  now:    { {k: settings.get(k) for k in changed} }\n"
                "Pass --out <new folder> for a separate run (or delete run.json if the change cannot affect the "
                "teacher's answers or which ones are accepted).")
        missing = sorted(set(settings) - set(before))
        if missing:  # recorded by an older version, which did not track these
            print(f"note: {path} does not record {', '.join(missing)} (older version); assuming they are unchanged",
                  flush=True)
            write_json(path, {**before, **settings})
        return
    if has_results and not path.exists():
        print(f"note: {out} has no run.json (made by an older version); assuming it used these settings", flush=True)
    write_json(path, settings)


def run_sync(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return asyncio.run(rejection_sample(*args, **kwargs))
