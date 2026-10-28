"""Recommendation evaluation through the generic `Task.evaluate` hook:
catalog-constrained beam search, then HR@K / NDCG@K (same as scripts/rec_eval_llm.py).

    python -m distillops eval --task tasks.rec.task:RecTask --task-arg category=Industrial_and_Scientific \
        --model <ckpt> --n 1000 --no-think
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Sequence

from distillops.io import write_json, write_jsonl

from .data import load_catalog
from .generate import build_trie, check_boundary, recommend_all
from .metrics import ranking_metrics


def evaluate_rec(
    task: Any,
    model: str,
    examples: Sequence[Any],
    out_dir: str | Path,
    enable_thinking: bool = True,
    beams: int = 20,
    batch_size: int = 0,
    chat_template: str | None = None,
    **_: Any,  # generic options that do not apply here (e.g. max_new_tokens)
) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM

    from distillops.chat import device, load_tokenizer
    from distillops.precision import compute_dtype

    tok = load_tokenizer(model, chat_template)
    lm = AutoModelForCausalLM.from_pretrained(model, dtype=compute_dtype(), device_map=device()).eval()
    titles = load_catalog(task.category).unique_titles()
    check_boundary(tok, titles)
    trie = build_trie(titles, tok)
    t0 = time.time()
    ranked_all, reasons_all = recommend_all(  # batch sizes 0 = auto from free GPU memory
        lm, tok, trie, [ex.inter.hist_titles for ex in examples], task.category_name, enable_thinking,
        num_beams=beams, beam_batch=batch_size,
    )
    secs = time.time() - t0

    m = ranking_metrics(ranked_all, [ex.inter.target_title for ex in examples])
    valid = set(titles)
    metrics = {**m, "task": task.name, "model": model, "n": len(examples), "beams": beams, "think": enable_thinking,
               "invalid_items": sum(t not in valid for r in ranked_all for t in r),
               "sec_per_example": round(secs / max(len(examples), 1), 3)}
    out = Path(out_dir)
    write_json(out / "metrics.json", metrics)
    write_jsonl(out / "predictions.jsonl", (
        {"id": task.example_id(ex), "target": ex.inter.target_title, "ranked": rk, "reasoning": rs}
        for ex, rk, rs in zip(examples, ranked_all, reasons_all)
    ))
    return metrics
