"""Greedy generation with a vision-language student (evaluation and GRPO validation).

Same contract as distillops.evaluate.generate, so it plugs into distillops.evaluate.evaluate_task
as `generate_fn`: (model, message_lists, **options) -> one completion per message list, in order.
"""

from __future__ import annotations

from typing import Any, Sequence

from . import images
from .model import (TemplateView, encode, float_inputs_to, load_model, load_processor, render, suppressed_ids,
                    tokenizer_of)


def generate_with(lm: Any, processor: Any, message_lists: Sequence[list[dict]], max_new_tokens: int = 512,
                  batch_size: int = 0, chat_template_kwargs: dict | None = None, max_pixels: int | None = None,
                  log: Any = print) -> list[str]:
    """Greedy completions from an already loaded model (GRPO validation reuses the training model)."""
    import torch

    from distillops.chat import stop_token_ids
    from distillops.memory import adaptive_chunks, max_sequences

    from .model import align_inputs

    align_inputs(processor, lm)
    tok = tokenizer_of(processor)
    side = tok.padding_side
    tok.padding_side = "left"  # decoder-only batching: prompts aligned at the right edge
    stops = stop_token_ids(TemplateView(processor), lm.generation_config.eos_token_id)
    banned = suppressed_ids(lm, processor)
    texts = [render(processor, m, add_generation_prompt=True, **(chat_template_kwargs or {})) for m in message_lists]
    parts = [images.image_parts(m) for m in message_lists]
    # most image-heavy, longest prompts first: an out-of-memory error shows up in the first batch
    order = sorted(range(len(texts)), key=lambda i: (-len(parts[i]), -len(texts[i])))
    dev = next(lm.parameters()).device
    if batch_size <= 0 and texts:
        first = order[0]
        longest = int(encode(processor, [texts[first]], [parts[first]], max_pixels)["input_ids"].shape[1])
        batch_size = max_sequences(lm, longest + max_new_tokens)
    log(f"start batch size: {batch_size}")
    out: list[str] = [""] * len(texts)
    # prompts with images and text-only prompts never share a batch: some processors refuse that
    groups = [[i for i in order if parts[i]], [i for i in order if not parts[i]]]

    group: list[int] = []

    def chunk(a: int, b: int) -> None:
        idx = group[a:b]
        enc = encode(processor, [texts[i] for i in idx], [parts[i] for i in idx], max_pixels, padding=True)
        enc = float_inputs_to(enc.to(dev), lm.dtype)
        with torch.no_grad():
            gen = lm.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id,
                              eos_token_id=stops, **({"suppress_tokens": banned} if banned else {}))
        for i, row in zip(idx, gen[:, enc["input_ids"].shape[1]:]):
            out[i] = tok.decode(row, skip_special_tokens=True)
        log(f"generated {b}/{len(group)}")

    try:
        for g in groups:
            if g:
                group[:] = g
                batch_size = adaptive_chunks(len(g), batch_size, chunk, log=log)
    finally:
        tok.padding_side = side
    return out


def generate_vision(model: str, message_lists: Sequence[list[dict]], max_new_tokens: int = 512, batch_size: int = 0,
                    chat_template_kwargs: dict | None = None, chat_template: str | None = None,
                    max_pixels: int | None = None, trust_remote_code: bool = False) -> list[str]:
    """Greedy completions for chat prompts that may contain images (order preserved).
    batch_size 0 = from free GPU memory and the model's size, shrinking on out-of-memory."""
    from distillops.chat import device
    from distillops.precision import compute_dtype

    processor = load_processor(model, chat_template, trust_remote_code)
    lm = load_model(model, dtype=compute_dtype(), device_map=device(), trust_remote_code=trust_remote_code).eval()
    return generate_with(lm, processor, message_lists, max_new_tokens, batch_size, chat_template_kwargs, max_pixels,
                         log=lambda s: print(s, flush=True))
