"""Task-agnostic student evaluation: generate with the student, score with the task.

    metrics = evaluate_task(task, "checkpoints/x/merged", examples, "results/eval/x")

Generation is plain batched greedy decoding with Hugging Face transformers (works
on any GPU, including Windows without vLLM). Tasks that need special decoding
(e.g. catalog-constrained beam search) override `Task.evaluate` instead.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Sequence

from .io import write_json, write_jsonl


def generate(
    model: str,
    message_lists: Sequence[list[dict[str, str]]],
    max_new_tokens: int = 512,
    batch_size: int = 0,
    chat_template_kwargs: dict[str, Any] | None = None,
    chat_template: str | None = None,
) -> list[str]:
    """Greedy completions for a list of chat prompts (order preserved).
    batch_size 0 = auto from free GPU memory and the model config (any GPU).
    chat_template_kwargs: options passed to the chat template (e.g. {"enable_thinking": False});
    templates ignore options they do not use.
    chat_template: see distillops/chat.py (None = the model's own, or a plain one if it has none)."""
    import torch
    from transformers import AutoModelForCausalLM

    from .chat import device, load_tokenizer

    tok = load_tokenizer(model, chat_template)
    tok.padding_side = "left"  # decoder-only batching: pad on the left so generation starts aligned
    from distillops.precision import compute_dtype

    dev = device()
    lm = AutoModelForCausalLM.from_pretrained(model, dtype=compute_dtype(), device_map=dev).eval()
    texts = [
        tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, **(chat_template_kwargs or {}))
        for m in message_lists
    ]
    # Longest prompts first: similar lengths share a batch (less padding), and an
    # out-of-memory error shows up in the first batch instead of after an hour.
    order = sorted(range(len(texts)), key=lambda i: -len(texts[i]))
    if batch_size <= 0 and texts:
        from .memory import max_sequences

        longest = len(tok(texts[order[0]], add_special_tokens=False)["input_ids"])
        batch_size = max_sequences(lm, longest + max_new_tokens)
    print(f"start batch size: {batch_size}", flush=True)
    out: list[str] = [""] * len(texts)

    def chunk(a: int, b: int) -> None:
        idx = order[a:b]
        enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
        with torch.no_grad():
            gen = lm.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
        for i, row in zip(idx, gen[:, enc["input_ids"].shape[1] :]):
            out[i] = tok.decode(row, skip_special_tokens=True)
        print(f"generated {b}/{len(order)}", flush=True)

    from .memory import adaptive_chunks  # shrinks the batch if a chunk nears the GPU's limit

    adaptive_chunks(len(order), batch_size, chunk)
    return out


def evaluate_task(
    task: Any,
    model: str,
    examples: Sequence[Any],
    out_dir: str | Path,
    generate_fn: Callable[..., list[str]] = generate,
    **gen_kwargs: Any,
) -> dict[str, Any]:
    t0 = time.time()
    completions = generate_fn(model, [task.student_messages(ex) for ex in examples], **gen_kwargs)
    scores = [task.score(ex, c) for ex, c in zip(examples, completions)]
    n = len(examples)
    metrics = {
        "task": task.name,
        "model": model,
        "n": n,
        "score": round(sum(scores) / n, 4) if n else 0.0,
        "sec_per_example": round((time.time() - t0) / max(n, 1), 3),
        **{k: v for k, v in gen_kwargs.items() if isinstance(v, (int, float, str, bool))},
    }
    if gen_kwargs.get("chat_template_kwargs"):
        metrics["chat_template_kwargs"] = gen_kwargs["chat_template_kwargs"]
    extra_metrics = getattr(task, "extra_metrics", None)  # dataset-level metrics (F1, BLEU, ...), optional
    extra = extra_metrics(examples, completions, scores) if callable(extra_metrics) else {}
    clash = sorted(set(extra) & set(metrics))
    if clash:
        raise ValueError(f"extra_metrics returned keys the framework already reports: {clash}")
    metrics.update(extra)
    out = Path(out_dir)
    write_json(out / "metrics.json", metrics)
    write_jsonl(out / "predictions.jsonl", (
        {"id": task.example_id(ex), "score": s, "completion": c} for ex, s, c in zip(examples, scores, completions)
    ))
    return metrics
