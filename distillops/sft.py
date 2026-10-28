"""Task-agnostic SFT on distilled data (TRL SFTTrainer + optional LoRA).

Input: JSONL rows in TRL conversational prompt/completion format
  {"prompt": [{"role": "user", ...}], "completion": [{"role": "assistant", ...}]}
TRL then computes the loss on completion tokens only (prompt is masked).

Key choices:
- completion-only loss;
- over-long rows are DROPPED, not truncated (truncation teaches "never finish");
- the student's chat template is the one used at inference (same tokenizer); models without one
  get a plain template (`chat_template`, see distillops/chat.py), saved with the trained model;
- LoRA on all linear layers with lr ~1e-4, or full fine-tuning with lr ~1e-5.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .chat import chat_ids


@dataclass
class SFTRunConfig:
    model: str
    train_file: str
    output_dir: str
    eval_file: str | None = None
    max_length: int = 1024
    epochs: float = 2.0
    lr: float = 1e-4
    batch_size: int = 8
    grad_accum: int = 4
    warmup_ratio: float = 0.03
    lora: bool = True
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    seed: int = 42
    logging_steps: int = 10
    save_strategy: str = "epoch"
    chat_template: str | None = None  # None = the model's own; "plain", a Jinja file or string (chat.py)
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path, **overrides: Any) -> "SFTRunConfig":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        raw.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**raw)


def _load_rows(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def length_filter(rows: list[dict], tokenizer, max_length: int) -> tuple[list[dict], dict]:
    """Drop rows whose prompt+completion exceed max_length tokens (never truncate). Also counts rows
    whose prompt tokens are not a prefix of prompt+completion tokens (the completion's first token
    merged with the prompt's last one): there the loss mask is off by a token."""
    kept, lengths, misaligned = [], [], 0
    for r in rows:
        prompt_ids = chat_ids(tokenizer, r["prompt"], add_generation_prompt=True)
        ids = chat_ids(tokenizer, r["prompt"] + r["completion"])
        misaligned += ids[: len(prompt_ids)] != prompt_ids
        lengths.append(len(ids))
        if len(ids) <= max_length:
            kept.append({"prompt": r["prompt"], "completion": r["completion"]})
    lengths.sort()
    stats = {
        "rows_in": len(rows),
        "rows_kept": len(kept),
        "dropped_too_long": len(rows) - len(kept),
        "prompt_boundary_mismatch": misaligned,
        "len_p50": lengths[len(lengths) // 2] if lengths else 0,
        "len_p99": lengths[int(len(lengths) * 0.99)] if lengths else 0,
        "len_max": lengths[-1] if lengths else 0,
    }
    return kept, stats


def show_masked_example(tokenizer, row: dict) -> str:
    """Render one training row exactly as the model sees it, marking the tokens that
    carry loss. Run this before every new training setup."""
    prompt_text = tokenizer.apply_chat_template(row["prompt"], tokenize=False, add_generation_prompt=True)
    full_text = tokenizer.apply_chat_template(row["prompt"] + row["completion"], tokenize=False)
    assert full_text.startswith(prompt_text), "chat template: prompt is not a prefix of prompt+completion"
    return f"[NO LOSS]{prompt_text}[/NO LOSS]\n[LOSS]{full_text[len(prompt_text):]}[/LOSS]"


def _warmup(ratio: float) -> dict:
    """transformers>=5 takes a ratio through warmup_steps (float in [0,1)); 4.x has warmup_ratio."""
    import transformers

    major = int(transformers.__version__.split(".")[0])
    return {"warmup_steps": ratio} if major >= 5 else {"warmup_ratio": ratio}


def train(cfg: SFTRunConfig) -> dict:
    import torch
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    from .chat import load_tokenizer

    tok = load_tokenizer(cfg.model, cfg.chat_template)
    rows, len_stats = length_filter(_load_rows(cfg.train_file), tok, cfg.max_length)
    print("length filter:", len_stats)
    if len_stats["prompt_boundary_mismatch"]:
        print(f"note: in {len_stats['prompt_boundary_mismatch']} rows the completion's first token merges with "
              "the end of the prompt (often a completion starting with whitespace); the loss mask is off by one "
              "token there. Strip leading whitespace from completions to avoid it.", flush=True)
    print(show_masked_example(tok, rows[0])[:2000])
    eval_ds = None
    if cfg.eval_file:
        eval_rows, _ = length_filter(_load_rows(cfg.eval_file), tok, cfg.max_length)
        eval_ds = Dataset.from_list(eval_rows)

    peft_config = None
    if cfg.lora:
        from peft import LoraConfig

        peft_config = LoraConfig(
            r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
            target_modules="all-linear", task_type="CAUSAL_LM",
        )

    from .precision import describe, train_settings

    prec = train_settings(full_finetune=not cfg.lora)
    defaults = {"gradient_checkpointing": True, "use_cpu": prec["use_cpu"]}
    defaults.update(cfg.extra)  # config `extra` may override defaults such as gradient_checkpointing
    print("precision:", describe(), flush=True)
    args = SFTConfig(
        **defaults,
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.lr,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        **_warmup(cfg.warmup_ratio),
        lr_scheduler_type="cosine",
        max_length=cfg.max_length,
        completion_only_loss=True,
        bf16=prec["bf16"],
        fp16=prec["fp16"],
        logging_steps=cfg.logging_steps,
        save_strategy=cfg.save_strategy,
        eval_strategy="epoch" if eval_ds is not None else "no",
        seed=cfg.seed,
        report_to="none",
        model_init_kwargs={"dtype": prec["model_dtype"]},  # fp32 if full fine-tuning
    )
    trainer = SFTTrainer(
        model=cfg.model, args=args, train_dataset=Dataset.from_list(rows), eval_dataset=eval_ds,
        processing_class=tok, peft_config=peft_config,
    )
    from .preflight import check_disk, model_bytes

    check_disk(cfg.output_dir, model_bytes(cfg.model) * 2, "SFT outputs (checkpoints + merged model)")
    result = trainer.train()
    trainer.save_model(cfg.output_dir)
    if cfg.lora:  # merged copy so evaluation/serving can load a plain HF model
        merged = trainer.model.merge_and_unload()
        merged.save_pretrained(Path(cfg.output_dir) / "merged")
        tok.save_pretrained(Path(cfg.output_dir) / "merged")
    return {"length_filter": len_stats, "train_loss": result.training_loss}
