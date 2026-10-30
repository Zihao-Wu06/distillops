"""SFT of a vision-language student on rows whose prompts contain images.

Same rules as distillops.sft (the YAML fields are a superset of SFTRunConfig):
- loss on the completion tokens only (prompt, image and padding tokens are masked);
- rows longer than max_length tokens, images included, are dropped, never truncated;
- the chat template, stop tokens and trained ids are saved with the model;
- LoRA is refused when the rows use template tokens the model never trained (LoRA cannot learn them).
Vision additions: the vision modules are frozen by default (train_vision), the projector trains
(train_projector), and LoRA adapts the language model only.

Training uses transformers' Trainer with a collator written here, so it does not depend on how
each TRL version handles vision models.
"""

from __future__ import annotations

import re

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from distillops.sft import SFT_FIELDS, SFTRunConfig, sft_disk_bytes

from . import images
from .model import (TemplateView, align_inputs, batches_mixed_ok, encode, freeze_vision, lengths, load_model,
                    load_processor, placeholder_ids, projector_modules, render, tokenizer_of, upcast_trainable,
                    vision_modules)


@dataclass
class VisionSFTConfig(SFTRunConfig):
    train_vision: bool = False  # train the image encoder too (more memory; rarely needed)
    train_projector: bool = True  # the layers that map image features into the language model
    vision_modules: list[str] | None = None  # regexes over module paths, to name the vision modules by hand
    trust_remote_code: bool = False  # run the model repository's own code (needed by a few models)


def length_filter(rows: list[dict], processor: Any, max_length: int, max_pixels: int | None = None
                  ) -> tuple[list[dict], dict]:
    """Rows that fit max_length, images included; each kept row gets `_prompt_len` (prompt tokens,
    masked from the loss). Rows are dropped, never truncated."""
    kept, lens, too_long, mismatch = [], [], 0, 0
    for row in rows:
        p_len, f_len, aligned = lengths(processor, row, max_pixels)
        lens.append(f_len)
        if f_len > max_length:
            too_long += 1
            continue
        mismatch += not aligned
        kept.append({**row, "_prompt_len": p_len})
    s = sorted(lens)
    stats = {"rows_in": len(rows), "rows_kept": len(kept), "dropped_too_long": too_long,
             "prompt_boundary_mismatch": mismatch,
             "len_p50": s[len(s) // 2] if s else 0, "len_p99": s[min(len(s) - 1, int(len(s) * 0.99))] if s else 0,
             "len_max": s[-1] if s else 0}
    return kept, stats


class Collator:
    """Batches of rows -> processor inputs with labels on the completion only."""

    def __init__(self, processor: Any, max_pixels: int | None = None):
        self.processor, self.max_pixels = processor, max_pixels
        self.tok = tokenizer_of(processor)

    def __call__(self, rows: list[dict]) -> dict:
        texts = [render(self.processor, r["prompt"] + r["completion"]) for r in rows]
        parts = [images.image_parts(r["prompt"]) for r in rows]
        side = self.tok.padding_side
        self.tok.padding_side = "right"  # prompt lengths count from the left edge
        try:
            batch = encode(self.processor, texts, parts, self.max_pixels, padding=True)
        finally:
            self.tok.padding_side = side
        labels = batch["input_ids"].clone()
        for i, r in enumerate(rows):
            labels[i, : r["_prompt_len"]] = -100
        labels[batch["attention_mask"] == 0] = -100
        batch["labels"] = labels
        return dict(batch)


def check_template_tokens(model: Any, processor: Any, row: dict, lora: bool) -> None:
    """distillops.sft.check_template_tokens for a vision model: the tokens the chat template writes
    must be trained for LoRA; image placeholders are skipped (their embeddings are replaced)."""
    from distillops.chat import untrained_token_ids

    view = TemplateView(processor)
    prompt = view.apply_chat_template(row["prompt"], add_generation_prompt=True)
    full = view.apply_chat_template(row["prompt"] + row["completion"])
    written = full[len(prompt):] if full[:len(prompt)] == prompt else full
    skip = placeholder_ids(model, processor)
    bad = [i for i in untrained_token_ids(model, view, [i for i in full if i not in skip], written) if i not in skip]
    if not bad:
        return
    names = ", ".join(view.convert_ids_to_tokens(bad))
    if lora:
        raise SystemExit(
            f"The training rows use special tokens whose embeddings are untrained in this model ({names}); LoRA "
            "does not train embeddings, so the model would never learn to write them (and would not stop at the "
            "end of its reply). Options: lora: false (full fine-tuning trains them; lr ~1e-5), or an already "
            "chat-trained (instruct) model.")
    print(f"note: {names} are untrained in this model; full fine-tuning will train them", flush=True)


def lora_config(model: Any, cfg: Any) -> Any:
    """LoRA on the language model's linear layers. The projector (train_projector) and the image
    encoder (train_vision) train fully next to it, as modules PEFT saves whole."""
    from peft import LoraConfig

    vis = vision_modules(model, cfg.vision_modules)
    proj = projector_modules(model)
    skip = vis + proj
    kwargs: dict[str, Any] = {}
    if skip:
        kwargs["exclude_modules"] = "|".join(r"(^|.*\.)" + re.escape(p) + r"(\..*)?$" for p in skip)
    full = (proj if cfg.train_projector else []) + (vis if cfg.train_vision else [])
    # a projector nested in the encoder is covered by the encoder's entry
    full = [m for m in full if not any(m.startswith(o + ".") for o in full if o != m)]
    if full:
        kwargs["modules_to_save"] = full
    return LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
                      target_modules="all-linear", task_type="CAUSAL_LM", **kwargs)


def _warmup(ratio: float) -> dict:
    import transformers

    return {"warmup_steps": ratio} if int(transformers.__version__.split(".")[0]) >= 5 else {"warmup_ratio": ratio}


def train(cfg: VisionSFTConfig) -> dict:
    import json

    from transformers import Trainer, TrainingArguments

    from distillops.chat import save_stop_tokens
    from distillops.precision import describe, train_settings
    from distillops.preflight import check_disk, param_count
    from distillops.task import write_trained_ids
    from distillops.trainargs import merge_extra

    processor = load_processor(cfg.model, cfg.chat_template, cfg.trust_remote_code)
    def read(path: str) -> list[dict]:
        with open(path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    rows, stats = length_filter(read(cfg.train_file), processor, cfg.max_length)
    print("length filter:", stats, flush=True)
    if not stats["rows_in"]:
        raise SystemExit(f"{cfg.train_file} has no rows")
    if not rows:
        raise SystemExit(f"no rows left in {cfg.train_file}: all {stats['rows_in']} are longer than "
                         f"max_length={cfg.max_length} tokens, images included (longest: {stats['len_max']}). "
                         "Raise max_length or lower the images' max_pixels.")
    if stats["prompt_boundary_mismatch"]:
        print(f"note: in {stats['prompt_boundary_mismatch']} rows the prompt is not a prefix of the full "
              "conversation as tokenized; the loss mask may be off by a token there", flush=True)
    eval_rows = None
    if cfg.eval_file:
        eval_rows, eval_stats = length_filter(read(cfg.eval_file), processor, cfg.max_length)
        if not eval_rows:
            raise SystemExit(f"no rows of {cfg.eval_file} fit max_length={cfg.max_length} tokens with their images")
    with_image = next((r for r in rows if images.has_images(r["prompt"])), None)
    text_only = next((r for r in rows if not images.has_images(r["prompt"])), None)
    if with_image and text_only and cfg.batch_size > 1 and \
            not batches_mixed_ok(processor, with_image["prompt"], text_only["prompt"]):
        raise SystemExit("the training rows mix prompts with images and text-only prompts, and this model's processor "
                         "cannot put both in one batch: set batch_size: 1 (raise grad_accum to keep the effective "
                         "batch), or train on one kind of row")

    prec = train_settings(full_finetune=not cfg.lora)
    print("precision:", describe(), flush=True)
    model = load_model(cfg.model, dtype=prec["model_dtype"], trust_remote_code=cfg.trust_remote_code)
    align_inputs(processor, model)
    check_template_tokens(model, processor, rows[0], cfg.lora)
    frozen = freeze_vision(model, cfg.train_vision, cfg.train_projector, cfg.vision_modules)
    print(f"vision modules: {frozen['vision_modules'] or 'none'}; projector: {frozen['projector_modules'] or 'none'}; "
          f"frozen parameters: {frozen['frozen_params']:,}", flush=True)
    if cfg.lora:
        from peft import get_peft_model

        model = get_peft_model(model, lora_config(model, cfg))
        cast = upcast_trainable(model)
        if cast:
            print(f"note: {cast:,} trainable parameters next to LoRA (e.g. the projector) moved to fp32", flush=True)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable parameters: {trainable:,}", flush=True)

    framework_args = dict(
        output_dir=cfg.output_dir, num_train_epochs=cfg.epochs, learning_rate=cfg.lr,
        per_device_train_batch_size=cfg.batch_size, gradient_accumulation_steps=cfg.grad_accum,
        **_warmup(cfg.warmup_ratio), lr_scheduler_type="cosine",
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=prec["bf16"], fp16=prec["fp16"], use_cpu=prec["use_cpu"],
        logging_steps=cfg.logging_steps, save_strategy=cfg.save_strategy, save_only_model=True, save_total_limit=1,
        seed=cfg.seed, report_to="none", remove_unused_columns=False,
        eval_strategy="epoch" if eval_rows else "no",
    )
    fields = {k: v for k, v in SFT_FIELDS.items() if k != "max_length"}  # the length filter applies max_length
    fields.update({k: "warmup_ratio" for k in _warmup(cfg.warmup_ratio)})
    try:
        args = TrainingArguments(**merge_extra(framework_args, cfg.extra, fields, "SFT"))
    except TypeError as e:  # e.g. an SFTConfig-only key in `extra`
        raise SystemExit(f"{e}. Vision SFT trains with transformers' Trainer, so `extra` takes TrainingArguments "
                         "fields (not TRL SFTConfig ones such as packing or model_init_kwargs).") from e
    need = sft_disk_bytes(param_count(cfg.model), lora=cfg.lora, args=args, n_rows=len(rows),
                          merged_bytes_per_param=prec["model_dtype"].itemsize)
    check_disk(cfg.output_dir, need, "SFT outputs (checkpoints + final model)")
    trainer = Trainer(model=model, args=args, train_dataset=rows, eval_dataset=eval_rows,
                      data_collator=Collator(processor))
    result = trainer.train()

    kept_ids = [(r.get("meta") or {}).get("id") for r in rows]
    known = all(i is not None for i in kept_ids)
    if not known:
        print("note: some training rows have no meta.id, so the trained examples are not recorded; GRPO cannot "
              "keep them out of its validation prompts", flush=True)
    main = trainer.is_world_process_zero()
    out_dirs = [Path(cfg.output_dir)]
    trainer.save_model(cfg.output_dir)
    if cfg.lora and main:  # merged copy so evaluation, GRPO and serving load a plain model
        merged = trainer.model.merge_and_unload()
        merged.save_pretrained(Path(cfg.output_dir) / "merged")
        out_dirs.append(Path(cfg.output_dir) / "merged")
    if main:
        view = TemplateView(processor)
        for d in out_dirs:
            processor.save_pretrained(d)
            save_stop_tokens(d, view)  # full fine-tuning: the model itself (an adapter folder is skipped)
            if known:
                write_trained_ids(d, [str(i) for i in kept_ids])
    return {"length_filter": stats, "train_loss": result.training_loss, "trainable_params": trainable,
            "vision": {k: v for k, v in frozen.items() if k != "frozen_params"}}
