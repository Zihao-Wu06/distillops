"""Task-agnostic GRPO stage (TRL GRPOTrainer), run after SFT.

What makes GRPO work in practice:
1. Start from the SFT checkpoint (cold start); RL from a raw small model rarely learns.
2. Keep only prompts with 0 < pass-rate < 1 under the SFT model: if all G samples
   are right (or all wrong) the group-normalised advantage is 0 and the prompt
   teaches nothing. `filter_by_pass_rate` does this.
3. Verifiable rewards supplied by the task (`Task.score`, the evaluation metric).
4. Log sample completions; watch reward, length, entropy and KL for hacking/collapse.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import yaml


@dataclass
class GRPORunConfig:
    model: str  # SFT checkpoint (merged)
    output_dir: str
    train_file: str | None = None  # JSONL rows {"prompt": [...], <reward columns>}; or built from a Task
    num_generations: int = 8
    max_prompt_length: int = 768
    max_completion_length: int = 256
    temperature: float = 1.0
    lr: float = 1e-6
    beta: float = 0.0  # KL coefficient (0 = no reference model, TRL default)
    epsilon: float = 0.2
    loss_type: str = "dapo"
    batch_size: int = 8
    grad_accum: int = 4
    max_steps: int = 500
    use_vllm: bool = True
    vllm_mode: str = "colocate"
    vllm_gpu_memory_utilization: float = 0.3
    lora: bool = False
    seed: int = 42
    logging_steps: int = 5
    save_steps: int = 100
    chat_template_kwargs: dict[str, Any] = field(default_factory=dict)  # e.g. {"enable_thinking": false}
    # Entropy-collapse guard, relative to the run's own starting entropy (entropy scales differ
    # across tasks, so no absolute threshold): stop when entropy stays below
    # entropy_floor_ratio x the first logged value for `entropy_patience` logs. 0 disables.
    entropy_floor_ratio: float = 0.1
    entropy_patience: int = 2
    # Validation-based checkpoint selection: every `eval_every` steps, generate greedily on
    # `eval_examples` held-out prompts and score them with Task.score; output_dir ends up holding the
    # best-scoring model (possibly the starting model). The task's "valid" split is used if it has one,
    # otherwise prompts are held out of training. eval_every = 0 disables it.
    eval_every: int = 25
    eval_examples: int = 200
    eval_patience: int = 0  # stop after this many checks without improvement (0 = never stop early)
    extra: dict[str, Any] = field(default_factory=dict)  # any other GRPOConfig argument

    @classmethod
    def from_yaml(cls, path: str | Path, **overrides: Any) -> "GRPORunConfig":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        raw.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**raw)


def filter_by_pass_rate(rows: Sequence[dict], pass_rates: Sequence[float], lo: float = 0.0, hi: float = 1.0) -> list[dict]:
    """Keep prompts whose pass-rate p (fraction of G samples rewarded) satisfies lo < p < hi."""
    return [r for r, p in zip(rows, pass_rates) if lo < p < hi]


def estimate_pass_rates(
    model: str,
    task: Any,
    examples: Sequence[Any],
    n: int = 8,
    temperature: float = 1.0,
    max_tokens: int = 256,
    chat_template_kwargs: dict[str, Any] | None = None,
    gpu_memory_utilization: float = 0.85,
) -> tuple[list[float], dict[str, Any]]:
    """Sample `n` completions per prompt with vLLM (the same sampling GRPO will use) and
    score them with the task. Returns per-example pass rates and summary statistics."""
    from vllm import LLM, SamplingParams

    from .memory import wait_for_free_memory

    wait_for_free_memory(gpu_memory_utilization + 0.05)  # the GPU may be shared with another job
    from .precision import compute_dtype, dtype_name

    llm = LLM(model=model, dtype=dtype_name(compute_dtype()), gpu_memory_utilization=gpu_memory_utilization, seed=0)
    tok = llm.get_tokenizer()
    prompts = [
        tok.apply_chat_template(task.student_messages(ex), tokenize=False, add_generation_prompt=True,
                                **(chat_template_kwargs or {}))
        for ex in examples
    ]
    outs = llm.generate(prompts, SamplingParams(n=n, temperature=temperature, max_tokens=max_tokens, seed=0))
    rates = [sum(task.score(ex, o.text) for o in out.outputs) / n for ex, out in zip(examples, outs)]
    hist = {"all_wrong": sum(r == 0 for r in rates), "mixed": sum(0 < r < 1 for r in rates),
            "all_right": sum(r == 1 for r in rates)}
    stats = {"examples": len(rates), "samples_per_example": n, "mean_pass_rate": round(sum(rates) / max(len(rates), 1), 4),
             **hist}
    del llm  # free GPU memory for training
    return rates, stats


def diagnose_pass_rates(stats: dict[str, Any], warn_below: float = 0.10) -> list[str]:
    """Plain-language advice from the pass-rate distribution (any task)."""
    n = max(stats["examples"], 1)
    mixed, wrong, right = stats["mixed"] / n, stats["all_wrong"] / n, stats["all_right"] / n
    notes = [f"{mixed:.1%} of prompts give a learning signal (some samples right, some wrong); "
             f"{wrong:.1%} are always wrong, {right:.1%} always right."]
    if mixed < warn_below:
        notes.append("The reward is sparse for this model: GRPO can only learn from the mixed prompts. "
                     "Options: more samples per prompt (--filter-samples, num_generations), more candidate prompts, "
                     "a stronger SFT start, or a denser Task.score that gives partial credit.")
    if right > 0.5:
        notes.append("Most prompts are already solved: GRPO has little left to teach; consider harder prompts.")
    return notes


def steps_for_epochs(n_prompts: int, epochs: float, cfg: "GRPORunConfig") -> int:
    """Optimizer steps that show each prompt `epochs` times (TRL batch sizes count completions)."""
    import math

    prompts_per_step = max(1, cfg.batch_size * cfg.grad_accum // cfg.num_generations)
    return max(1, math.ceil(n_prompts * epochs / prompts_per_step))


def fit_micro_batch(cfg: "GRPORunConfig", total_bytes: int | None = None, vocab: int | None = None,
                    train_state_bytes: int | None = None) -> tuple[int, int]:
    """Largest per-device batch (completions) whose loss-time logits fit in memory, with gradient
    accumulation raised so batch x accumulation, and therefore the optimisation, stays unchanged.

    The loss needs fp32 logits over the vocabulary for every completion token (plus a log-softmax
    copy): completions x max_completion_length x vocab x 8 bytes. 64 math solutions of 512 tokens
    need ~20GB; 64 product titles of 96 tokens need ~4GB, so a fixed batch fits one task and not
    the other."""
    total = cfg.batch_size * cfg.grad_accum
    if total_bytes is None or vocab is None or train_state_bytes is None:
        try:
            import torch
            from transformers import AutoConfig

            from .preflight import model_bytes

            if not torch.cuda.is_available():
                return cfg.batch_size, cfg.grad_accum
            total_bytes = total_bytes or torch.cuda.get_device_properties(0).total_memory
            c = AutoConfig.from_pretrained(cfg.model)
            vocab = vocab or getattr(getattr(c, "text_config", c), "vocab_size")
            params = model_bytes(cfg.model) / 2  # weight files are 16-bit -> parameter count
            # fp32 weights + grads + Adam (16 B/param) for full fine-tuning; a LoRA base needs ~2 B/param
            train_state_bytes = train_state_bytes or int(params * (2 if cfg.lora else 16))
        except Exception:
            return cfg.batch_size, cfg.grad_accum
    vllm = cfg.vllm_gpu_memory_utilization if (cfg.use_vllm and cfg.vllm_mode == "colocate") else 0.0
    budget = 0.5 * (total_bytes * (1 - vllm) - train_state_bytes)  # half for logits, half for activations
    per_completion = cfg.max_completion_length * vocab * 8
    cap = max(1, int(budget // per_completion))
    batch = max(d for d in range(1, total + 1) if total % d == 0 and d <= max(cap, 1))
    batch = min(batch, cfg.batch_size)
    if batch != cfg.batch_size:
        print(f"micro-batch {cfg.batch_size} -> {batch} completions (x{total // batch} accumulation, same "
              f"{total} per step): {cfg.max_completion_length}-token completions x {vocab} vocab", flush=True)
    return batch, total // batch


def entropy_guard(floor_ratio: float, patience: int):
    """TrainerCallback that stops training on entropy collapse: entropy below
    `floor_ratio` x its first logged value for `patience` consecutive logs."""
    from transformers import TrainerCallback

    class EntropyGuard(TrainerCallback):
        def __init__(self) -> None:
            self.initial: float | None = None
            self.low = 0
            self.stopped_at: int | None = None

        def on_log(self, args, state, control, logs=None, **kwargs):
            ent = (logs or {}).get("entropy")
            if ent is None or floor_ratio <= 0:
                return
            if self.initial is None:
                self.initial = ent
                return
            floor = floor_ratio * self.initial
            self.low = self.low + 1 if ent < floor else 0
            if self.low >= patience:
                self.stopped_at = state.global_step
                print(f"ENTROPY COLLAPSE: entropy {ent:.4f} < {floor:.4f} ({floor_ratio:.0%} of the starting "
                      f"{self.initial:.4f}) for {patience} logs at step {state.global_step}; stopping. The policy now "
                      "repeats one answer per prompt and is likely memorising them. Options: fewer epochs, "
                      "beta > 0 (KL to the start model), a lower learning rate, more prompts.", flush=True)
                control.should_training_stop = True

    return EntropyGuard()


def split_validation(task: Any, examples: Sequence[Any], n: int, seed: int = 0) -> tuple[list[Any], list[Any], str]:
    """(training examples, validation examples, source). Uses the task's "valid" split when it
    exists; otherwise holds `n` examples out of `examples` and removes them from training."""
    import random

    if n <= 0:
        return list(examples), [], "none"
    try:
        valid = list(task.load_examples("valid"))
    except Exception:  # most tasks without a validation split raise here (e.g. datasets: unknown split)
        valid = []
    if valid:
        pick = random.Random(seed).sample(valid, min(n, len(valid)))
        return list(examples), pick, "valid split"
    if len(examples) < 2:
        return list(examples), [], "none"
    k = min(n, len(examples) // 5)  # never hold out more than 20% of the training prompts
    held = set(random.Random(seed + 1).sample(range(len(examples)), k))
    return ([ex for i, ex in enumerate(examples) if i not in held],
            [ex for i, ex in enumerate(examples) if i in held], "held out of training")


def make_evaluator(task: Any, examples: Sequence[Any], max_new_tokens: int,
                   render: Callable[[list[dict]], str] | None = None, batch_size: int = 0) -> Callable[[Any, Any], float]:
    """Returns evaluate(model, tokenizer) -> mean Task.score of greedy completions on `examples`.
    Runs on the model being trained (no extra copy); switches it to eval mode and back."""

    def evaluate(model: Any, tok: Any) -> float:
        import torch

        from .memory import adaptive_chunks, max_sequences
        from .precision import compute_dtype

        texts = [render(task.student_messages(ex)) if render else
                 tok.apply_chat_template(task.student_messages(ex), tokenize=False, add_generation_prompt=True)
                 for ex in examples]
        order = sorted(range(len(texts)), key=lambda i: -len(texts[i]))
        outs: list[str] = [""] * len(texts)
        old_side, was_training = tok.padding_side, model.training
        tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        device = next(model.parameters()).device
        use_amp = device.type == "cuda"
        if use_amp:  # drop blocks cached by the last training step, so the sizing sees what eval really needs
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        if batch_size > 0:
            start = batch_size
        else:
            longest = len(tok(texts[order[0]], add_special_tokens=False)["input_ids"]) if texts else 1
            start = max_sequences(model, longest + max_new_tokens)

        def chunk(a: int, b: int) -> None:
            idx = order[a:b]
            enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(device)
            with torch.no_grad(), torch.autocast(device.type, dtype=compute_dtype(), enabled=use_amp):
                gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                     pad_token_id=tok.pad_token_id, use_cache=True)
            for i, row in zip(idx, gen[:, enc["input_ids"].shape[1]:]):
                outs[i] = tok.decode(row, skip_special_tokens=True)

        try:
            model.eval()
            adaptive_chunks(len(texts), start, chunk, log=lambda m: print("validation:", m, flush=True))
        finally:
            tok.padding_side = old_side
            if was_training:
                model.train()
        return sum(float(task.score(ex, c)) for ex, c in zip(examples, outs)) / max(len(examples), 1)

    return evaluate


def validation_selector(evaluate: Callable[[Any], float], every: int, patience: int, save_best: Callable[[], None]):
    """TrainerCallback: score the model before training and every `every` steps (and at the end);
    call `save_best()` whenever a trained step beats every earlier score, including the start."""
    from transformers import TrainerCallback

    class ValidationSelector(TrainerCallback):
        def __init__(self) -> None:
            self.history: list[tuple[int, float]] = []
            self.best: float | None = None
            self.best_step: int | None = None
            self.bad = 0

        def _check(self, step: int, model: Any) -> None:
            if self.history and self.history[-1][0] == step:  # already scored this step
                return
            score = evaluate(model)
            self.history.append((step, score))
            if self.best is None or score > self.best:
                self.best, self.best_step, self.bad = score, step, 0
                if step > 0:
                    save_best()
                note = "best so far" + (" (starting model)" if step == 0 else ", saved")
            else:
                self.bad += 1
                note = f"no improvement ({self.bad} in a row; best {self.best:.4f} at step {self.best_step})"
            print(f"validation score at step {step}: {score:.4f} -- {note}", flush=True)

        def on_train_begin(self, args, state, control, model=None, **kwargs):
            if every > 0:
                self._check(0, model)

        def on_step_end(self, args, state, control, model=None, **kwargs):
            if every > 0 and state.global_step % every == 0:
                self._check(state.global_step, model)
                if patience > 0 and self.bad >= patience:
                    print(f"validation did not improve for {patience} checks: stopping early", flush=True)
                    control.should_training_stop = True

        def on_train_end(self, args, state, control, model=None, **kwargs):
            if every > 0:
                self._check(state.global_step, model)

    return ValidationSelector()


def _snapshot(model: Any, n: int = 8) -> dict[str, Any]:
    """CPU copies of a few trainable weight matrices, to measure how much training moved them."""
    named = [(k, p) for k, p in model.named_parameters() if p.requires_grad and p.dim() == 2]
    step = max(1, len(named) // n)
    return {k: p.detach().float().cpu().clone() for k, p in named[::step][:n]}


def _relative_change(before: dict[str, Any], model: Any) -> float:
    now = dict(model.named_parameters())
    rel = [((now[k].detach().float().cpu() - v).norm() / (v.norm() + 1e-12)).item() for k, v in before.items() if k in now]
    return sum(rel) / max(len(rel), 1)


def _text(completion: Any) -> str:
    if isinstance(completion, list):  # conversational: [{"role": "assistant", "content": ...}]
        return completion[0]["content"]
    return completion


def task_rows_and_reward(
    task: Any, examples: Sequence[Any], render: Callable[[list[dict]], str] | None = None
) -> tuple[list[dict], Callable]:
    """GRPO data and reward for any Task: prompts from `student_messages`, reward from
    `score` (the same function used for evaluation, so RL optimises the reported metric).
    TRL passes dataset columns to reward functions as keyword arguments, so each row
    carries its example id and the reward looks the example up."""
    by_id = {task.example_id(ex): ex for ex in examples}
    # `render` turns chat messages into the final prompt string (e.g. with enable_thinking=False);
    # without it the prompt stays conversational and TRL applies the chat template.
    rows = [{"prompt": render(m) if render else m, "ex_id": task.example_id(ex)}
            for ex, m in ((ex, task.student_messages(ex)) for ex in examples)]

    def reward(completions, ex_id, **kwargs) -> list[float]:
        return [float(task.score(by_id[i], _text(c))) for c, i in zip(completions, ex_id)]

    reward.__name__ = f"{task.name}_score"  # TRL logs rewards under the function name
    return rows, reward


def train(
    cfg: GRPORunConfig,
    reward_funcs: Sequence[Callable],
    reward_weights: Sequence[float] | None = None,
    rows: Sequence[dict] | None = None,
    evaluator: Callable[[Any, Any], float] | None = None,
) -> dict:
    import shutil

    import torch
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    if cfg.use_vllm and cfg.vllm_mode == "colocate":
        from .memory import wait_for_free_memory

        wait_for_free_memory(cfg.vllm_gpu_memory_utilization + 0.15)  # vLLM share + room for training
    if rows is None:
        with open(cfg.train_file, encoding="utf-8") as f:
            rows = [json.loads(l) for l in f if l.strip()]
    peft_config = None
    if cfg.lora:
        from peft import LoraConfig

        peft_config = LoraConfig(r=32, lora_alpha=64, target_modules="all-linear", task_type="CAUSAL_LM")

    from .precision import describe, train_settings

    prec = train_settings(full_finetune=not cfg.lora)
    print("precision:", describe(), flush=True)
    batch, accum = fit_micro_batch(cfg)
    args = GRPOConfig(
        output_dir=cfg.output_dir,
        learning_rate=cfg.lr,
        per_device_train_batch_size=batch,
        gradient_accumulation_steps=accum,
        num_generations=cfg.num_generations,
        max_prompt_length=cfg.max_prompt_length,
        max_completion_length=cfg.max_completion_length,
        temperature=cfg.temperature,
        beta=cfg.beta,
        epsilon=cfg.epsilon,
        loss_type=cfg.loss_type,
        max_steps=cfg.max_steps,
        use_vllm=cfg.use_vllm,
        vllm_mode=cfg.vllm_mode,
        vllm_gpu_memory_utilization=cfg.vllm_gpu_memory_utilization,
        bf16=prec["bf16"],
        fp16=prec["fp16"],
        gradient_checkpointing=True,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        log_completions=True,  # inspect real outputs: the first defence against reward hacking
        # Full fine-tuning checkpoints with optimizer state are ~5x the model size (a 1.7B run
        # filled a 50GB disk); RL runs are short, so keep weights only and one checkpoint.
        **{"save_only_model": True, "save_total_limit": 1, **{k: v for k, v in cfg.extra.items()
                                                              if k in ("save_only_model", "save_total_limit")}},
        seed=cfg.seed,
        report_to="none",
        # fp32 master weights for full fine-tuning, low-precision compute (distillops/precision.py)
        model_init_kwargs={"dtype": prec["model_dtype"]},
        reward_weights=list(reward_weights) if reward_weights else None,
        **{k: v for k, v in cfg.extra.items() if k not in ("save_only_model", "save_total_limit")},
    )
    guard = entropy_guard(cfg.entropy_floor_ratio, cfg.entropy_patience)
    callbacks = [guard]
    selector = None
    holder: dict[str, Any] = {}  # the trainer, once built (the callback needs its model and tokenizer)
    if evaluator is not None and cfg.eval_every > 0:
        selector = validation_selector(
            lambda model: evaluator(model, holder["trainer"].processing_class),
            cfg.eval_every, cfg.eval_patience, lambda: holder["trainer"].save_model(cfg.output_dir))
        callbacks.append(selector)
    trainer = GRPOTrainer(
        model=cfg.model, reward_funcs=list(reward_funcs), args=args,
        train_dataset=Dataset.from_list(rows), peft_config=peft_config, callbacks=callbacks,
    )
    holder["trainer"] = trainer
    from .preflight import check_disk, min_expected_change, model_bytes

    # weights-only checkpoints in the training dtype: (save_total_limit + final) copies
    per_copy = model_bytes(cfg.model) * (2 if prec["model_dtype"] == torch.float32 else 1)
    # periodic checkpoints + the final/best model (+ the last model when validation picks the best)
    copies = (args.save_total_limit or 1) + 1 + (1 if selector else 0)
    check_disk(cfg.output_dir, per_copy * copies, "GRPO checkpoints")
    before = _snapshot(trainer.model)
    result = trainer.train()
    change = _relative_change(before, trainer.model)
    rms = sum(v.std().item() for v in before.values()) / max(len(before), 1)
    floor = min_expected_change(cfg.lr, result.global_step, rms)
    print(f"weight change (relative, sampled tensors): {change:.2e} (expected at least {floor:.1e})", flush=True)
    if change < floor:
        print("WARNING: the weights barely changed. Check precision (low-precision master weights round "
              "small updates away), the learning rate, and the share of prompts with a learning signal.", flush=True)
    out = {"train_loss": result.training_loss, "rows": len(rows), "weight_change": change,
           "steps": result.global_step, "stopped_for_entropy_collapse_at": guard.stopped_at}
    if selector is None:
        trainer.save_model(cfg.output_dir)
        return out
    # output_dir holds the best model; the last one goes to output_dir/last for reference
    trainer.save_model(str(Path(cfg.output_dir) / "last"))
    if selector.best_step == 0:  # nothing beat the starting model: deliver the starting model
        if Path(cfg.model).is_dir():
            for f in Path(cfg.model).iterdir():
                if f.is_file():
                    shutil.copy2(f, Path(cfg.output_dir) / f.name)
            note = f"no trained step beat the starting model; {cfg.output_dir} holds a copy of {cfg.model}"
        else:
            note = f"no trained step beat the starting model ({cfg.model}); use it instead of {cfg.output_dir}"
    else:
        note = f"best validation score at step {selector.best_step}; saved in {cfg.output_dir}"
    print("validation:", note, flush=True)
    out.update({"validation_history": selector.history, "best_step": selector.best_step,
                "best_validation_score": selector.best, "validation_note": note})
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    (Path(cfg.output_dir) / "validation.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out
