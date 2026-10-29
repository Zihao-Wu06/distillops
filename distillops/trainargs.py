"""Merging a config's `extra` into the trainer arguments the framework builds (SFT and GRPO).

`extra` may set any TRL argument, including ones the framework sets itself (report_to,
lr_scheduler_type, gradient_checkpointing, ...): the user's value wins and is reported. Dict-valued
arguments are merged key by key, so e.g. `model_init_kwargs: {attn_implementation: ...}` keeps the
dtype the framework chose.
Arguments that have their own config field (e.g. learning_rate <- lr) must be set through that
field, because the framework also uses the field (length filter, step counts, batch fitting).
"""

from __future__ import annotations

from typing import Any


def merge_extra(args: dict[str, Any], extra: dict[str, Any], fields: dict[str, str], what: str) -> dict[str, Any]:
    """`args` updated with `extra`. `fields` maps trainer argument -> config field that sets it."""
    clash = {k: fields[k] for k in extra if k in fields}
    if clash:
        hint = ", ".join(f"{k} -> `{v}`" for k, v in clash.items())
        raise SystemExit(f"{what}: set these through their config fields, not `extra`: {hint}")
    merged = dict(args)
    for k, v in extra.items():
        merged[k] = {**args[k], **v} if isinstance(args.get(k), dict) and isinstance(v, dict) else v
    overridden = sorted(k for k in extra if k in args and args[k] != merged[k])
    if overridden:
        print(f"{what}: `extra` overrides the framework's default for {', '.join(overridden)}", flush=True)
    return merged
