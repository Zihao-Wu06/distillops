"""Task-agnostic GRPO stage (TRL GRPOTrainer), run after SFT.

What makes GRPO work in practice:
1. Start from the SFT checkpoint (cold start); RL from a raw small model rarely learns.
2. Keep only prompts whose G samples differ in score under the SFT model: if all
   scores are equal (all right, all wrong, or the same partial credit) the
   group-normalised advantage is 0 and the prompt teaches nothing. `has_signal`.
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
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.0
    seed: int = 42
    logging_steps: int = 5
    save_steps: int = 100
    chat_template_kwargs: dict[str, Any] = field(default_factory=dict)  # e.g. {"enable_thinking": false}
    chat_template: str | None = None  # None = the model's own; "plain", a Jinja file or string (chat.py)
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


def has_signal(scores: Sequence[float], tol: float = 1e-9) -> bool:
    """True if a prompt's sampled completions differ in score. GRPO's advantage is each score minus
    the group mean, so a group whose scores are all equal (all 0, all 1, or the same partial
    credit) teaches nothing. Works for binary and partial-credit rewards alike."""
    return bool(scores) and max(scores) - min(scores) > tol


def filter_by_signal(rows: Sequence[Any], scores: Sequence[Sequence[float]]) -> list[Any]:
    """Keep the rows whose sampled scores differ (`has_signal`)."""
    return [r for r, s in zip(rows, scores) if has_signal(s)]


def signal_stats(scores: Sequence[Sequence[float]]) -> dict[str, Any]:
    """How many prompts carry a learning signal, and why the others do not."""
    flat = [x for s in scores for x in s]
    same = [s[0] for s in scores if s and not has_signal(s)]
    return {"examples": len(scores), "samples_per_example": len(scores[0]) if scores else 0,
            "mean_score": round(sum(flat) / len(flat), 4) if flat else 0.0,
            "mixed": sum(has_signal(s) for s in scores),
            "all_wrong": sum(v <= 0 for v in same), "all_right": sum(v >= 1 for v in same),
            "all_same_partial": sum(0 < v < 1 for v in same)}


def estimate_pass_rates(
    model: str,
    task: Any,
    examples: Sequence[Any],
    n: int = 8,
    temperature: float = 1.0,
    max_tokens: int = 256,
    chat_template_kwargs: dict[str, Any] | None = None,
    gpu_memory_utilization: float = 0.85,
    chat_template: str | None = None,
    max_prompt_length: int | None = None,
) -> tuple[list[list[float]], dict[str, Any]]:
    """Sample `n` completions per prompt with vLLM, the way GRPO will (same temperature, completion
    limit, prompt truncation and context length), and score them with the task. Returns each
    example's scores and summary statistics (`signal_stats`)."""
    from vllm import LLM, SamplingParams

    from .memory import require_free_memory

    require_free_memory(gpu_memory_utilization, "--filter-gpu-memory")
    from .precision import compute_dtype, dtype_name

    # Like TRL's GRPO: context = prompt + completion limit (otherwise vLLM reserves the model's full
    # context, which may not fit), and over-long prompts keep their last max_prompt_length tokens.
    max_model_len = max_prompt_length + max_tokens if max_prompt_length else None
    llm = LLM(model=model, dtype=dtype_name(compute_dtype()), gpu_memory_utilization=gpu_memory_utilization,
              max_model_len=max_model_len, seed=0)
    from .chat import prepare_tokenizer

    tok = prepare_tokenizer(llm.get_tokenizer(), chat_template, model)
    from .chat import end_of_turn_id

    eot = end_of_turn_id(tok)  # stop where the template ends a reply, as GRPO and evaluation do
    prompts = [
        tok.apply_chat_template(task.student_messages(ex), tokenize=False, add_generation_prompt=True,
                                **(chat_template_kwargs or {}))
        for ex in examples
    ]
    params = SamplingParams(n=n, temperature=temperature, max_tokens=max_tokens, seed=0,
                            truncate_prompt_tokens=max_prompt_length, stop_token_ids=[eot] if eot is not None else None)
    outs = llm.generate(prompts, params)
    scores = [[float(task.score(ex, o.text)) for o in out.outputs] for ex, out in zip(examples, outs)]
    del llm  # free GPU memory for training
    return scores, signal_stats(scores)


def diagnose_pass_rates(stats: dict[str, Any], warn_below: float = 0.10) -> list[str]:
    """Plain-language advice from the score distribution (any task, binary or partial-credit score)."""
    n = max(stats["examples"], 1)
    mixed, wrong, right = stats["mixed"] / n, stats["all_wrong"] / n, stats["all_right"] / n
    partial = stats.get("all_same_partial", 0) / n
    notes = [f"{mixed:.1%} of prompts give a learning signal (their samples differ in score); "
             f"{wrong:.1%} always score 0, {right:.1%} always score 1"
             + (f", {partial:.1%} get the same partial score every time." if partial else ".")]
    if mixed < warn_below:
        notes.append("The reward is sparse for this model: GRPO can only learn from prompts whose samples differ. "
                     "Options: more samples per prompt (--filter-samples, num_generations), more candidate prompts, "
                     "a stronger SFT start, or a Task.score that separates better from worse answers.")
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

    The loss needs fp32 values over the vocabulary for every completion token, three times over: the
    logits, their log-softmax, and the logits' gradient in the backward pass: completions x
    max_completion_length x vocab x 12 bytes. 64 math solutions of 512 tokens need ~60GB; 64 product
    titles of 96 tokens ~11GB, so a fixed batch fits one task and not the other."""
    total = cfg.batch_size * cfg.grad_accum
    if total_bytes is None or vocab is None or train_state_bytes is None:
        try:
            import torch
            from transformers import AutoConfig

            from .memory import model_dims
            from .preflight import param_count

            if not torch.cuda.is_available():
                return cfg.batch_size, cfg.grad_accum
            total_bytes = total_bytes or torch.cuda.get_device_properties(0).total_memory
            vocab = vocab or model_dims(AutoConfig.from_pretrained(cfg.model))["vocab"]
            params = param_count(cfg.model)
            if not params:
                return cfg.batch_size, cfg.grad_accum
            # fp32 weights + grads + Adam (16 B/param) for full fine-tuning; a LoRA base needs ~2 B/param
            train_state_bytes = train_state_bytes or int(params * (2 if cfg.lora else 16))
        except Exception as e:  # unknown architecture, no network, ...: keep the configured batch
            print(f"note: could not size the GRPO micro-batch ({e}); using batch_size={cfg.batch_size}", flush=True)
            return cfg.batch_size, cfg.grad_accum
    vllm = cfg.vllm_gpu_memory_utilization if (cfg.use_vllm and cfg.vllm_mode == "colocate") else 0.0
    free_for_training = total_bytes * (1 - vllm) - train_state_bytes
    if free_for_training <= 0:
        print(f"WARNING: the model's training state (~{train_state_bytes / 2**30:.1f} GB"
              + (", full fine-tuning at 16 B/param" if not cfg.lora else "") + ") does not fit next to vLLM's "
              f"share ({vllm:.0%}) on this {total_bytes / 2**30:.0f} GB GPU. Options: lora: true, a smaller model, "
              "or a lower vllm_gpu_memory_utilization.", flush=True)
    budget = 0.5 * free_for_training  # half for logits, half for activations
    per_completion = cfg.max_completion_length * vocab * 12  # fp32 logits, log-softmax and their gradient
    cap = max(1, int(budget // per_completion))
    batch = max(d for d in range(1, total + 1) if total % d == 0 and d <= max(cap, 1))
    batch = min(batch, cfg.batch_size)
    if batch != cfg.batch_size:
        print(f"micro-batch {cfg.batch_size} -> {batch} completions (x{total // batch} accumulation, same "
              f"{total} per step): {cfg.max_completion_length}-token completions x {vocab} vocab", flush=True)
    return batch, total // batch


# trainer arguments that have their own config field (see distillops/trainargs.py)
GRPO_FIELDS = {"output_dir": "output_dir", "learning_rate": "lr", "per_device_train_batch_size": "batch_size",
               "gradient_accumulation_steps": "grad_accum", "num_generations": "num_generations",
               "max_prompt_length": "max_prompt_length", "max_completion_length": "max_completion_length",
               "temperature": "temperature", "beta": "beta", "epsilon": "epsilon", "loss_type": "loss_type",
               "max_steps": "max_steps", "use_vllm": "use_vllm", "vllm_mode": "vllm_mode",
               "vllm_gpu_memory_utilization": "vllm_gpu_memory_utilization", "logging_steps": "logging_steps",
               "save_steps": "save_steps", "seed": "seed"}


def prompt_length_args(config_cls: Any, rows: Sequence[dict], tok: Any, max_prompt_length: int | None,
                       max_completion_length: int) -> tuple[dict[str, Any], list[dict]]:
    """Trainer arguments for the prompt limit, and the rows to train on.

    TRL < 0.28 takes `max_prompt_length`: it keeps the last max_prompt_length tokens of a longer
    prompt and sizes a colocated vLLM to prompt + completion. Later TRL removed the argument and
    never truncates, so the limit is kept here: vLLM gets the same context (`vllm_max_model_length`)
    and longer prompts, which would not fit it, are left out (with a count)."""
    import dataclasses

    fields = {f.name for f in dataclasses.fields(config_cls)}
    if "max_prompt_length" in fields:
        return {"max_prompt_length": max_prompt_length}, list(rows)
    if not max_prompt_length:
        return {}, list(rows)
    from .chat import chat_ids

    def length(prompt: Any) -> int:  # as TRL tokenizes it: text as is, messages with the chat template
        if isinstance(prompt, str):
            return len(tok(prompt)["input_ids"])
        return len(chat_ids(tok, prompt, add_generation_prompt=True))

    kept = [r for r in rows if length(r["prompt"]) <= max_prompt_length]
    if len(kept) < len(rows):
        print(f"note: this TRL version does not truncate prompts; {len(rows) - len(kept)} of {len(rows)} "
              f"prompts are longer than max_prompt_length ({max_prompt_length} tokens) and are left out",
              flush=True)
    if not kept:
        raise SystemExit(f"GRPO: every prompt is longer than max_prompt_length ({max_prompt_length} tokens); "
                         "raise it")
    args = {}
    if "vllm_max_model_length" in fields:
        args["vllm_max_model_length"] = max_prompt_length + max_completion_length
    return args, kept


def grpo_disk_bytes(params: int, lora: bool, args: Any, validation: bool, bytes_per_param: int = 4,
                    start_bytes: int = 0) -> int:
    """Disk a GRPO run writes. Full fine-tuning: each checkpoint and saved model holds the weights in
    the training dtype (fp32), plus Adam's moments (8 B/param) unless save_only_model; there are
    save_total_limit checkpoints, the final or best model, and output_dir/last with validation.
    LoRA saves adapters (small), but output_dir may receive a copy of the start model when no
    trained step beats it (`start_bytes`: its weight files' size; 0 = assume 16-bit)."""
    if not params:
        return 0
    start_copy = start_bytes or params * 2  # the start model, copied when nothing beats it
    if lora:
        return start_copy if validation else 0
    import math

    per_ckpt = bytes_per_param + (0 if args.save_only_model else 8)
    every = args.save_steps if args.save_steps >= 1 else math.ceil(args.max_steps * args.save_steps)
    n_ckpt = args.save_total_limit or max(1, args.max_steps // max(int(every), 1))
    if str(getattr(args.save_strategy, "value", args.save_strategy)) == "no":
        n_ckpt = 0
    models = (2 if validation else 1) * bytes_per_param
    return params * (n_ckpt * per_ckpt + models)


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


def split_validation(task: Any, examples: Sequence[Any], n: int, seed: int = 0,
                     exclude_ids: set[str] | None = None) -> tuple[list[Any], list[Any], str]:
    """(training examples, validation examples, source). Uses the task's "valid" split when it
    exists; otherwise holds `n` examples out of `examples` and removes them from training.
    `exclude_ids`: examples the start model was trained on. They are never used for validation:
    a model scores high on prompts it has learned the answers to, so such a score cannot show
    whether GRPO helps (validation leakage). With none left, validation is off ("none: ...")."""
    import random

    if n <= 0:
        return list(examples), [], "none"
    seen = exclude_ids or set()
    try:
        valid = list(task.load_examples("valid"))
    except Exception:  # most tasks without a validation split raise here (e.g. datasets: unknown split)
        valid = []
    if valid:
        unseen = [ex for ex in valid if task.example_id(ex) not in seen]
        if not unseen:
            return list(examples), [], "none: every prompt of the valid split was in the start model's training data"
        pick = random.Random(seed).sample(unseen, min(n, len(unseen)))
        note = f", {len(valid) - len(unseen)} seen in training left out" if len(unseen) < len(valid) else ""
        return list(examples), pick, f"valid split{note}"
    candidates = [i for i, ex in enumerate(examples) if task.example_id(ex) not in seen]
    if len(examples) < 2 or not candidates:
        reason = ("every prompt was in the start model's training data" if len(examples) >= 2 and seen
                  else "too few prompts")
        return list(examples), [], f"none: {reason}"
    k = min(n, len(examples) // 5, len(candidates))  # never hold out more than 20% of the training prompts
    held = set(random.Random(seed + 1).sample(candidates, k))
    note = (" (prompts the start model was trained on left out; the rest, e.g. ones the teacher failed, may be "
            "harder than average: a 'valid' split is more representative)") if len(candidates) < len(examples) else ""
    return ([ex for i, ex in enumerate(examples) if i not in held],
            [ex for i, ex in enumerate(examples) if i in held], f"held out of training{note}")


def make_evaluator(task: Any, examples: Sequence[Any], max_new_tokens: int,
                   render: Callable[[list[dict]], str] | None = None,
                   batch_size: int = 0) -> Callable[[Any, Any], tuple[float, float]]:
    """Returns evaluate(model, tokenizer) -> (mean Task.score of greedy completions on `examples`,
    its standard error). Runs on the model being trained (no extra copy); switches it to eval mode
    and back."""

    def evaluate(model: Any, tok: Any) -> tuple[float, float]:
        import torch

        from .chat import stop_token_ids
        from .memory import adaptive_chunks, max_sequences
        from .precision import compute_dtype

        stops = stop_token_ids(tok, getattr(getattr(model, "generation_config", None), "eos_token_id", None))
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
                                     pad_token_id=tok.pad_token_id, eos_token_id=stops, use_cache=True)
            for i, row in zip(idx, gen[:, enc["input_ids"].shape[1]:]):
                outs[i] = tok.decode(row, skip_special_tokens=True)

        try:
            model.eval()
            adaptive_chunks(len(texts), start, chunk, log=lambda m: print("validation:", m, flush=True))
        finally:
            tok.padding_side = old_side
            if was_training:
                model.train()
            if use_amp:  # give generation's cached blocks back before training resumes (fragmentation)
                torch.cuda.empty_cache()
        scores = [float(task.score(ex, c)) for ex, c in zip(examples, outs)]
        return mean_and_se(scores)

    return evaluate


def mean_and_se(scores: Sequence[float]) -> tuple[float, float]:
    """Mean and its standard error: differences between checks smaller than ~2 SE are noise."""
    n = len(scores)
    if not n:
        return 0.0, 0.0
    mean = sum(scores) / n
    var = sum((s - mean) ** 2 for s in scores) / (n - 1) if n > 1 else 0.0
    return mean, (var / n) ** 0.5


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
            result = evaluate(model)
            score, se = result if isinstance(result, tuple) else (result, None)
            self.history.append((step, score))
            if self.best is None or score > self.best:
                self.best, self.best_step, self.bad = score, step, 0
                if step > 0:
                    save_best()
                note = "best so far" + (" (starting model)" if step == 0 else ", saved")
            else:
                self.bad += 1
                note = f"no improvement ({self.bad} in a row; best {self.best:.4f} at step {self.best_step})"
            spread = f" ± {se:.4f}" if se is not None else ""
            print(f"validation score at step {step}: {score:.4f}{spread} -- {note}", flush=True)

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
    """CPU copies of a few trainable weight matrices, to measure how much training moved them.
    Matrices that start at zero (LoRA's B) are skipped: a change relative to zero is meaningless."""
    named = [(k, p) for k, p in model.named_parameters()
             if p.requires_grad and p.dim() == 2 and bool(p.detach().abs().sum() > 0)]
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
    from .task import check_ids

    by_id = dict(zip(check_ids(task, examples), examples))
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

    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    if cfg.use_vllm and cfg.vllm_mode == "colocate":
        from .memory import require_free_memory

        require_free_memory(cfg.vllm_gpu_memory_utilization, "vllm_gpu_memory_utilization")
    if rows is None:
        with open(cfg.train_file, encoding="utf-8") as f:
            rows = [json.loads(l) for l in f if l.strip()]
    peft_config = None
    if cfg.lora:
        from peft import LoraConfig

        peft_config = LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
                                 target_modules="all-linear", task_type="CAUSAL_LM")

    from .precision import describe, train_settings

    prec = train_settings(full_finetune=not cfg.lora)
    print("precision:", describe(), flush=True)
    batch, accum = fit_micro_batch(cfg)
    from .chat import end_of_turn_id, load_tokenizer, save_stop_tokens, stop_token_ids
    from .trainargs import merge_extra

    tok = load_tokenizer(cfg.model, cfg.chat_template)
    tok.padding_side = "left"  # TRL's GRPO pads prompts on the left
    stops = stop_token_ids(tok)
    eot = end_of_turn_id(tok)
    if eot is not None:  # TRL ends (and masks) completions at the tokenizer's EOS: make that the end of a turn
        print(f"note: the chat template ends a reply with {tok.convert_ids_to_tokens(eot)}, not EOS "
              f"{tok.eos_token}; generation stops at both", flush=True)
        tok.eos_token = tok.convert_ids_to_tokens(eot)
    length_args, rows = prompt_length_args(GRPOConfig, rows, tok, cfg.max_prompt_length, cfg.max_completion_length)

    framework_args = dict(
        output_dir=cfg.output_dir,
        learning_rate=cfg.lr,
        per_device_train_batch_size=batch,
        gradient_accumulation_steps=accum,
        num_generations=cfg.num_generations,
        **length_args,
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
        # Full fine-tuning checkpoints with optimizer state are several times the model size; RL
        # runs are short, so keep weights only and one checkpoint.
        save_only_model=True,
        save_total_limit=1,
        use_cpu=prec["use_cpu"],
        seed=cfg.seed,
        report_to="none",
        # fp32 master weights for full fine-tuning, low-precision compute (distillops/precision.py)
        model_init_kwargs={"dtype": prec["model_dtype"]},
        reward_weights=list(reward_weights) if reward_weights else None,
        # rollouts stop at EOS or the end-of-turn token (vLLM SamplingParams / transformers GenerationConfig).
        # transformers pads a row that stopped early; TRL masks after the first `tok.eos_token` (now the
        # end-of-turn token), so pad with that token, or a reply ending at the old EOS would count its padding
        generation_kwargs=({"stop_token_ids": stops} if cfg.use_vllm else
                           {"eos_token_id": stops, **({"pad_token_id": eot} if eot is not None else {})}),
    )
    args = GRPOConfig(**merge_extra(framework_args, cfg.extra, GRPO_FIELDS, "GRPO"))
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
        model=cfg.model, reward_funcs=list(reward_funcs), args=args, processing_class=tok,
        train_dataset=Dataset.from_list(rows), peft_config=peft_config, callbacks=callbacks,
    )
    holder["trainer"] = trainer
    from .preflight import check_disk, min_expected_change, param_count, stored_bytes

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
            print("WARNING: the weights barely changed. Check precision (low-precision master weights round "
                  "small updates away), the learning rate, and the share of prompts with a learning signal.",
                  flush=True)
    else:
        print("note: no trainable weight matrix to compare; the weight-change check is skipped", flush=True)
    out = {"train_loss": result.training_loss, "rows": len(rows), "weight_change": change,
           "steps": result.global_step, "stopped_for_entropy_collapse_at": guard.stopped_at}
    main = trainer.is_world_process_zero()  # one process writes files under a multi-process launch
    from .task import read_trained_ids, write_trained_ids

    start_ids = read_trained_ids(cfg.model)
    if start_ids is None and not Path(cfg.model).is_dir():
        start_ids = set()  # a hub model: not trained on this task's rows by distillops
    # what the trained model has seen; unknown (None) when the start model's own record is missing
    trained_ids = None if start_ids is None else start_ids | {str(r["ex_id"]) for r in rows if "ex_id" in r}
    if selector is None:
        trainer.save_model(cfg.output_dir)
        if main:
            save_stop_tokens(cfg.output_dir, tok)
            if trained_ids is not None:
                write_trained_ids(cfg.output_dir, trained_ids)
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
    for d in (cfg.output_dir, Path(cfg.output_dir) / "last"):
        if main:
            save_stop_tokens(d, tok)
    if main and trained_ids is not None:  # output_dir holds the start model when nothing beat it
        write_trained_ids(Path(cfg.output_dir) / "last", trained_ids)
        write_trained_ids(cfg.output_dir, start_ids if selector.best_step == 0 else trained_ids)
    out.update({"validation_history": selector.history, "best_step": selector.best_step,
                "best_validation_score": selector.best, "validation_note": note})
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    (Path(cfg.output_dir) / "validation.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out
