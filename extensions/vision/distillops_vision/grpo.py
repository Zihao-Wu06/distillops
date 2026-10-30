"""GRPO for a vision-language student, reward = Task.score.

Same behaviour as distillops.grpo (the YAML fields are a superset of GRPORunConfig): the pass-rate
filter, validation on prompts the start model never trained on, the entropy guard, the weight-change
check, stop tokens and trained ids saved with the model. Vision differences:
- prompts are rendered by the processor's chat template (image placeholders included) and the
  images go to TRL in an `images` column, sized by each image part's max_pixels;
- prompts longer than max_prompt_length, images included, are left out, so a prompt is never
  truncated through its image tokens;
- the vision modules are frozen by default (train_vision), as in vision SFT.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from distillops.grpo import (GRPO_FIELDS, GRPORunConfig, entropy_guard, fit_micro_batch, grpo_disk_bytes, mean_and_se,
                             signal_stats, task_rows_and_reward, validation_selector)

from . import images
from .generate import generate_with
from .model import (TemplateView, align_inputs, freeze_vision, load_config, load_model, load_processor, prompt_tokens,
                    render, suppressed_ids, tokenizer_of, upcast_trainable)


@dataclass
class VisionGRPOConfig(GRPORunConfig):
    train_vision: bool = False
    train_projector: bool = True
    vision_modules: list[str] | None = None
    trust_remote_code: bool = False


def fit_prompts(task: Any, examples: Sequence[Any], processor: Any, max_prompt_length: int | None,
                max_pixels: int | None, template_kwargs: dict | None = None) -> tuple[list[Any], dict]:
    """Examples whose prompt, images included, fits max_prompt_length (the rest are left out)."""
    if not max_prompt_length:
        return list(examples), {"prompts": len(examples), "too_long": 0}
    kept, longest = [], 0
    for ex in examples:
        n = prompt_tokens(processor, task.student_messages(ex), max_pixels, **(template_kwargs or {}))
        longest = max(longest, n)
        if n <= max_prompt_length:
            kept.append(ex)
    stats = {"prompts": len(examples), "too_long": len(examples) - len(kept), "longest": longest}
    if stats["too_long"]:
        print(f"note: {stats['too_long']} of {len(examples)} prompts are longer than max_prompt_length="
              f"{max_prompt_length} tokens with their images (longest {longest}) and are left out; raise "
              "max_prompt_length or lower the images' max_pixels to keep them", flush=True)
    return kept, stats


def rows_and_reward(task: Any, examples: Sequence[Any], processor: Any, max_pixels: int | None,
                    template_kwargs: dict | None = None) -> tuple[list[dict], Callable]:
    """GRPO rows (rendered prompt with image placeholders, the images, the example id) and the
    reward (Task.score), built on distillops' task_rows_and_reward."""
    rows, reward = task_rows_and_reward(
        task, examples, lambda m: render(processor, m, add_generation_prompt=True, **(template_kwargs or {})))
    for row, ex in zip(rows, examples):  # rows follow the examples' order
        row["images"] = [images.load_part(p, max_pixels) for p in images.image_parts(task.student_messages(ex))]
    return rows, reward


def estimate_pass_rates(model: str, task: Any, examples: Sequence[Any], processor: Any, n: int = 8,
                        temperature: float = 1.0, max_tokens: int = 256, template_kwargs: dict | None = None,
                        gpu_memory_utilization: float = 0.85, max_prompt_length: int | None = None,
                        max_pixels: int | None = None, trust_remote_code: bool = False) -> tuple[list[list[float]], dict]:
    """distillops.grpo.estimate_pass_rates with images: `n` samples per prompt with vLLM, scored by the task."""
    from vllm import LLM, SamplingParams

    from distillops.chat import end_of_turn_id
    from distillops.memory import require_free_memory
    from distillops.precision import compute_dtype, dtype_name

    require_free_memory(gpu_memory_utilization, "--filter-gpu-memory")
    msgs = [task.student_messages(ex) for ex in examples]
    most = max((len(images.image_parts(m)) for m in msgs), default=0)
    llm = LLM(model=model, dtype=dtype_name(compute_dtype()), gpu_memory_utilization=gpu_memory_utilization,
              max_model_len=(max_prompt_length + max_tokens) if max_prompt_length else None, seed=0,
              trust_remote_code=trust_remote_code, **({"limit_mm_per_prompt": {"image": most}} if most else {}))
    eot = end_of_turn_id(TemplateView(processor))
    inputs = []
    for m in msgs:
        item: dict[str, Any] = {"prompt": render(processor, m, add_generation_prompt=True, **(template_kwargs or {}))}
        parts = images.image_parts(m)
        if parts:
            item["multi_modal_data"] = {"image": [images.load_part(p, max_pixels) for p in parts]}
        inputs.append(item)
    from .compat import vllm_ban

    banned = suppressed_ids(load_config(model, trust_remote_code), processor)
    params = SamplingParams(n=n, temperature=temperature, max_tokens=max_tokens, seed=0,
                            stop_token_ids=[eot] if eot is not None else None, **vllm_ban(banned))
    outs = llm.generate(inputs, params)
    scores = [[float(task.score(ex, o.text)) for o in out.outputs] for ex, out in zip(examples, outs)]
    return scores, signal_stats(scores)


def make_evaluator(task: Any, examples: Sequence[Any], max_new_tokens: int, max_pixels: int | None,
                   template_kwargs: dict | None = None) -> Callable[[Any, Any], tuple[float, float]]:
    """evaluate(model, processor) -> (mean Task.score of greedy completions, standard error), on the
    model being trained (switched to eval mode and back)."""

    def evaluate(model: Any, processor: Any) -> tuple[float, float]:
        import torch

        was_training = model.training
        cuda = next(model.parameters()).device.type == "cuda"
        if cuda:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        from distillops.precision import compute_dtype

        try:
            model.eval()
            # low-precision compute on the GPU, as distillops' validation (the weights may be fp32)
            with torch.autocast("cuda", dtype=compute_dtype(), enabled=cuda):
                outs = generate_with(model, processor, [task.student_messages(ex) for ex in examples], max_new_tokens,
                                     0, template_kwargs, max_pixels, log=lambda m: print("validation:", m, flush=True))
        finally:
            if was_training:
                model.train()
            if cuda:
                torch.cuda.empty_cache()
        return mean_and_se([float(task.score(ex, c)) for ex, c in zip(examples, outs)])

    return evaluate


def rollout_kwargs(use_vllm: bool, stops: list[int], eot: int | None, banned: list[int]) -> dict[str, Any]:
    """Generation settings for GRPO rollouts: stop at EOS or the end of turn, never sample `banned`.
    vLLM takes SamplingParams fields; transformers takes GenerationConfig fields, and pads a row that
    stopped early with the end-of-turn token (TRL masks after the first one)."""
    from .compat import vllm_ban

    if use_vllm:
        return {"stop_token_ids": stops, **vllm_ban(banned)}
    return {"eos_token_id": stops, **({"pad_token_id": eot} if eot is not None else {}),
            **({"suppress_tokens": banned} if banned else {})}


def train(cfg: VisionGRPOConfig, reward_funcs: Sequence[Callable], rows: Sequence[dict],
          evaluator: Callable[[Any, Any], Any] | None = None, processor: Any = None) -> dict:
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    from distillops.chat import end_of_turn_id, save_stop_tokens, stop_token_ids
    from distillops.grpo import _relative_change, _snapshot
    from distillops.precision import describe, train_settings
    from distillops.preflight import check_disk, min_expected_change, param_count, stored_bytes
    from distillops.task import read_trained_ids, write_trained_ids
    from distillops.trainargs import merge_extra

    from .sft import lora_config

    from .compat import require_grpo_support

    require_grpo_support(cfg.use_vllm, list(getattr(load_config(cfg.model, cfg.trust_remote_code), "architectures",
                                                    None) or []))
    if cfg.use_vllm and cfg.vllm_mode == "colocate":
        from distillops.memory import require_free_memory

        require_free_memory(cfg.vllm_gpu_memory_utilization, "vllm_gpu_memory_utilization")
    processor = processor or load_processor(cfg.model, cfg.chat_template, cfg.trust_remote_code)
    tok = tokenizer_of(processor)
    tok.padding_side = "left"  # TRL's GRPO pads prompts on the left
    view = TemplateView(processor)
    stops = stop_token_ids(view)
    eot = end_of_turn_id(view)
    if eot is not None:  # TRL ends (and masks) completions at the tokenizer's EOS: make that the end of a turn
        print(f"note: the chat template ends a reply with {tok.convert_ids_to_tokens(eot)}, not EOS {tok.eos_token}; "
              "generation stops at both", flush=True)
        tok.eos_token = tok.convert_ids_to_tokens(eot)

    prec = train_settings(full_finetune=not cfg.lora)
    print("precision:", describe(), flush=True)
    model = load_model(cfg.model, dtype=prec["model_dtype"], trust_remote_code=cfg.trust_remote_code)
    align_inputs(processor, model)
    first = next((r for r in rows if r.get("images")), None)
    if first is not None:  # what TRL cannot handle fails deep inside training: check it now
        from .compat import require_grpo_inputs

        probe = processor(text=[first["prompt"]], images=[first["images"]], return_tensors="pt",
                          add_special_tokens=False)
        require_grpo_inputs(set(probe.keys()), [len(r.get("images") or []) for r in rows])
    banned = suppressed_ids(model, processor)  # never sampled: an image placeholder breaks the scoring step
    frozen = freeze_vision(model, cfg.train_vision, cfg.train_projector, cfg.vision_modules)
    print(f"vision modules: {frozen['vision_modules'] or 'none'}; projector: {frozen['projector_modules'] or 'none'}; "
          f"frozen parameters: {frozen['frozen_params']:,}", flush=True)
    peft_config = lora_config(model, cfg) if cfg.lora else None
    batch, accum = fit_micro_batch(cfg)
    framework_args = dict(
        output_dir=cfg.output_dir, learning_rate=cfg.lr, per_device_train_batch_size=batch,
        gradient_accumulation_steps=accum, num_generations=cfg.num_generations,
        max_prompt_length=cfg.max_prompt_length, max_completion_length=cfg.max_completion_length,
        temperature=cfg.temperature, beta=cfg.beta, epsilon=cfg.epsilon, loss_type=cfg.loss_type,
        max_steps=cfg.max_steps, use_vllm=cfg.use_vllm, vllm_mode=cfg.vllm_mode,
        vllm_gpu_memory_utilization=cfg.vllm_gpu_memory_utilization, bf16=prec["bf16"], fp16=prec["fp16"],
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=cfg.logging_steps, save_steps=cfg.save_steps, log_completions=True,
        save_only_model=True, save_total_limit=1, use_cpu=prec["use_cpu"], seed=cfg.seed, report_to="none",
        generation_kwargs=rollout_kwargs(cfg.use_vllm, stops, eot, banned),
    )
    args = GRPOConfig(**merge_extra(framework_args, cfg.extra, GRPO_FIELDS, "GRPO"))
    guard = entropy_guard(cfg.entropy_floor_ratio, cfg.entropy_patience)
    callbacks: list = [guard]
    selector = None
    holder: dict[str, Any] = {}
    if evaluator is not None and cfg.eval_every > 0:
        selector = validation_selector(lambda m: evaluator(m, processor), cfg.eval_every, cfg.eval_patience,
                                       lambda: holder["trainer"].save_model(cfg.output_dir))
        callbacks.append(selector)
    trainer = GRPOTrainer(model=model, reward_funcs=list(reward_funcs), args=args, processing_class=processor,
                          train_dataset=Dataset.from_list(list(rows)), peft_config=peft_config, callbacks=callbacks)
    holder["trainer"] = trainer
    if cfg.lora:
        cast = upcast_trainable(trainer.model)
        if cast:
            print(f"note: {cast:,} trainable parameters next to LoRA (e.g. the projector) moved to fp32", flush=True)
    need = grpo_disk_bytes(param_count(cfg.model), lora=cfg.lora, args=args, validation=selector is not None,
                           bytes_per_param=prec["model_dtype"].itemsize, start_bytes=stored_bytes(cfg.model))
    check_disk(cfg.output_dir, need, "GRPO checkpoints and outputs")
    before = _snapshot(trainer.model)
    result = trainer.train()
    change = None
    if before:
        change = _relative_change(before, trainer.model)
        rms = sum(v.std().item() for v in before.values()) / len(before)
        floor = min_expected_change(cfg.lr, result.global_step, rms)
        print(f"weight change (relative, sampled tensors): {change:.2e} (expected at least {floor:.1e})", flush=True)
        if change < floor:
            print("WARNING: the weights barely changed. Check precision, the learning rate, and the share of "
                  "prompts with a learning signal.", flush=True)
    out = {"train_loss": result.training_loss, "rows": len(rows), "weight_change": change,
           "steps": result.global_step, "stopped_for_entropy_collapse_at": guard.stopped_at}
    main = trainer.is_world_process_zero()
    start_ids = read_trained_ids(cfg.model)
    if start_ids is None and not Path(cfg.model).is_dir():
        start_ids = set()
    trained_ids = None if start_ids is None else start_ids | {str(r["ex_id"]) for r in rows if "ex_id" in r}

    def finish(d: str | Path, ids: Any) -> None:  # trainer.save_model already wrote the processor
        if main:
            save_stop_tokens(d, view)
            if ids is not None:
                write_trained_ids(d, ids)

    if selector is None:
        trainer.save_model(cfg.output_dir)
        finish(cfg.output_dir, trained_ids)
        return out
    last = Path(cfg.output_dir) / "last"
    trainer.save_model(str(last))
    finish(last, trained_ids)
    if selector.best_step == 0:  # nothing beat the starting model: deliver the starting model
        if Path(cfg.model).is_dir():
            for f in Path(cfg.model).iterdir():
                if f.is_file():
                    shutil.copy2(f, Path(cfg.output_dir) / f.name)
            note = f"no trained step beat the starting model; {cfg.output_dir} holds a copy of {cfg.model}"
        else:
            note = f"no trained step beat the starting model ({cfg.model}); use it instead of {cfg.output_dir}"
        finish(cfg.output_dir, start_ids)
    else:
        note = f"best validation score at step {selector.best_step}; saved in {cfg.output_dir}"
        finish(cfg.output_dir, trained_ids)
    print("validation:", note, flush=True)
    out.update({"validation_history": selector.history, "best_step": selector.best_step,
                "best_validation_score": selector.best, "validation_note": note})
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    (Path(cfg.output_dir) / "validation.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out
