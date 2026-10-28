"""One command line for every task.

    python -m distillops teacher --task tasks.math.gsm8k:GSM8KTask --teacher <teacher.yaml> --n 100
    python -m distillops labels  --task tasks.math.gsm8k:GSM8KTask                  # label-only control data
    python -m distillops sft     configs/sft_qwen3.yaml --train_file <accepted.jsonl>
    python -m distillops eval    --task tasks.math.gsm8k:GSM8KTask --model <ckpt> --n 500
    python -m distillops grpo    configs/grpo_gsm8k_qwen3_0.6b.yaml --task tasks.math.gsm8k:GSM8KTask --filter-samples 8
    python -m distillops merge   --base Qwen/Qwen3-1.7B --adapter <lora_dir> --out <merged_dir>
    python -m distillops cost    --teacher <teacher.yaml>                           # teacher spend so far
    python -m distillops doctor                                                       # check this machine first

`--task` is `module:Class` or a YAML file with `task:` and `args:`. Constructor
arguments can also be given as repeated `--task-arg key=value`.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import random
from pathlib import Path
from typing import Any

import yaml

CHAT_TEMPLATE_HELP = ("'plain', a Jinja file or a Jinja string; default: the model's own template, "
                      "or 'plain' if it has none (see distillops/chat.py)")


def load_task(spec: str, task_args: list[str] | None = None) -> Any:
    kwargs: dict[str, Any] = {}
    if spec.endswith((".yaml", ".yml")):
        cfg = yaml.safe_load(Path(spec).read_text(encoding="utf-8"))
        spec, kwargs = cfg["task"], dict(cfg.get("args") or {})
    for kv in task_args or []:
        key, _, value = kv.partition("=")
        kwargs[key] = yaml.safe_load(value)  # "3" -> 3, "true" -> True, "abc" -> "abc"
    module, _, cls = spec.partition(":")
    if not cls:
        raise SystemExit(f"--task must look like module.path:ClassName (got {spec!r})")
    return getattr(importlib.import_module(module), cls)(**kwargs)


def subsample(examples: list[Any], n: int, seed: int) -> list[Any]:
    return examples if not n or n >= len(examples) else random.Random(seed).sample(examples, n)


def _task_args(p: argparse.ArgumentParser, split: str) -> None:
    p.add_argument("--task", required=True, help="module:Class or task YAML")
    p.add_argument("--task-arg", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--split", default=split)
    p.add_argument("--n", type=int, default=0, help="random subset size (0 = all)")
    p.add_argument("--seed", type=int, default=0)


def cmd_teacher(a: argparse.Namespace) -> None:
    from .rejection import rejection_sample
    from .teacher import Teacher, TeacherConfig

    tcfg = TeacherConfig.from_yaml(a.teacher)
    task = load_task(a.task, a.task_arg)
    examples = subsample(task.load_examples(a.split), a.n, a.seed)
    out = Path(a.out or f"results/{task.name}/teacher_{a.split}_n{a.n}")
    teacher = Teacher(tcfg)
    stats = asyncio.run(rejection_sample(task, teacher, examples, out, a.max_attempts))
    print(stats)
    print(f"SFT data: {out / 'accepted.jsonl'}")


def cmd_labels(a: argparse.Namespace) -> None:
    from .io import write_jsonl

    task = load_task(a.task, a.task_arg)
    examples = subsample(task.load_examples(a.split), a.n, a.seed)
    out = Path(a.out or f"data/sft/{task.name}/labels_{a.split}_n{a.n}.jsonl")
    print(f"{write_jsonl(out, (task.gold_sft_example(ex) for ex in examples))} rows -> {out}")


def cmd_sft(a: argparse.Namespace) -> None:
    from .io import write_json
    from .sft import SFTRunConfig, train

    cfg = SFTRunConfig.from_yaml(a.config, train_file=a.train_file, output_dir=a.output_dir, model=a.model,
                                 epochs=a.epochs, chat_template=a.chat_template)
    res = train(cfg)
    write_json(Path(cfg.output_dir) / "sft_summary.json", res)
    print(res)


def cmd_eval(a: argparse.Namespace) -> None:
    task = load_task(a.task, a.task_arg)
    examples = subsample(task.load_examples(a.split), a.n, a.seed)
    out = Path(a.out or f"results/{task.name}/eval/{a.tag or Path(a.model).name}")
    gen = {"max_new_tokens": a.max_new_tokens, "batch_size": a.batch, "enable_thinking": not a.no_think}
    if a.chat_template:  # only when set: tasks that override evaluate() need not accept it
        gen["chat_template"] = a.chat_template
    metrics = task.evaluate(a.model, examples, out, **gen)
    print(metrics)


def _exclude(task: Any, examples: list[Any], path: str | None) -> list[Any]:
    """Drop examples whose id appears as meta.id in a JSONL file (e.g. the rows a checkpoint
    was fine-tuned on, so RL / distillation sees new data)."""
    if not path:
        return examples
    from .io import read_jsonl

    seen = {r.get("meta", {}).get("id") for r in read_jsonl(path)}
    return [ex for ex in examples if task.example_id(ex) not in seen]


def cmd_grpo(a: argparse.Namespace) -> None:
    import json

    from .grpo import (GRPORunConfig, diagnose_pass_rates, estimate_pass_rates, make_evaluator, split_validation,
                       steps_for_epochs, task_rows_and_reward, train)
    from .io import write_json

    cfg = GRPORunConfig.from_yaml(a.config, model=a.model, chat_template=a.chat_template)
    task = load_task(a.task, a.task_arg)
    if a.max_steps:
        cfg.max_steps = a.max_steps
    if a.output_dir:
        cfg.output_dir = a.output_dir
    if a.eval_every is not None:
        cfg.eval_every = a.eval_every
    if a.eval_examples is not None:
        cfg.eval_examples = a.eval_examples
    examples = subsample(_exclude(task, task.load_examples(a.split), a.exclude), a.n, a.seed)
    valid: list = []
    if cfg.eval_every > 0 and cfg.eval_examples > 0 and not a.filter_only:
        # chosen before --keep-ids / filtering, deterministically, so held-out prompts never reach training
        examples, valid, source = split_validation(task, examples, cfg.eval_examples, a.seed)
        print(f"validation: {len(valid)} prompts ({source}), checked every {cfg.eval_every} steps", flush=True)
    if a.keep_ids:  # prompts kept by an earlier --filter-only run
        keep = set(Path(a.keep_ids).read_text(encoding="utf-8").split())
        examples = [ex for ex in examples if task.example_id(ex) in keep]
        print(f"{len(examples)} prompts from {a.keep_ids}", flush=True)
    if a.filter_samples:
        rates, stats = estimate_pass_rates(cfg.model, task, examples, n=a.filter_samples, temperature=cfg.temperature,
                                           max_tokens=cfg.max_completion_length,
                                           chat_template_kwargs=cfg.chat_template_kwargs,
                                           gpu_memory_utilization=a.filter_gpu_memory,
                                           chat_template=cfg.chat_template)
        examples = [ex for ex, r in zip(examples, rates) if 0 < r < 1]
        stats["advice"] = diagnose_pass_rates(stats)
        print("pass-rate filter:", json.dumps(stats), f"-> {len(examples)} prompts kept", flush=True)
        for note in stats["advice"]:
            print("  *", note, flush=True)
        write_json(Path(cfg.output_dir) / "pass_rate_filter.json", stats)
        (Path(cfg.output_dir) / "pass_rate_kept.txt").write_text(
            "\n".join(task.example_id(ex) for ex in examples), encoding="utf-8")
        if a.filter_only:  # run training in a fresh process: vLLM may not release all GPU memory
            return
    if a.epochs:
        cfg.max_steps = steps_for_epochs(len(examples), a.epochs, cfg)
        print(f"{len(examples)} prompts x {a.epochs} epochs -> {cfg.max_steps} steps", flush=True)
    render = None
    if cfg.chat_template_kwargs:  # render prompts ourselves: works whatever TRL version is installed
        from .chat import load_tokenizer

        tok = load_tokenizer(cfg.model, cfg.chat_template)
        render = lambda m: tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True,  # noqa: E731
                                                   **cfg.chat_template_kwargs)
    rows, reward = task_rows_and_reward(task, examples, render)
    evaluator = make_evaluator(task, valid, cfg.max_completion_length, render) if valid else None
    print(train(cfg, [reward], rows=rows, evaluator=evaluator))


def cmd_merge(a: argparse.Namespace) -> None:
    """Merge a LoRA adapter into its base model (e.g. after copying only the adapter to a new machine)."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .precision import compute_dtype

    base = AutoModelForCausalLM.from_pretrained(a.base, dtype=compute_dtype())
    merged = PeftModel.from_pretrained(base, a.adapter).merge_and_unload()
    merged.save_pretrained(a.out)
    AutoTokenizer.from_pretrained(a.adapter).save_pretrained(a.out)
    print(f"merged {a.adapter} into {a.base} -> {a.out}")


def cmd_cost(a: argparse.Namespace) -> None:
    import json
    import sqlite3

    from .teacher import TeacherConfig

    cfg = TeacherConfig.from_yaml(a.teacher)
    n = prompt = hit = out = 0
    for (v,) in sqlite3.connect(cfg.cache_path).execute("SELECT v FROM kv"):
        r = json.loads(v)
        n, prompt, hit, out = n + 1, prompt + r["prompt_tokens"], hit + r["cache_hit_tokens"], out + r["completion_tokens"]
    p = cfg.price
    cost = ((prompt - hit) * p.input_cache_miss + hit * p.input_cache_hit + out * p.output) / 1e6
    print(f"calls {n:,} | input {prompt:,} (cache hit {hit:,}) | output {out:,} | cost {cost:,.4f} "
          f"(in the currency of `price` in {a.teacher})")


def cmd_doctor(a: argparse.Namespace) -> None:
    from .preflight import doctor

    doctor()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m distillops", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("teacher", help="rejection-sample teacher outputs into SFT data")
    _task_args(p, "train")
    p.add_argument("--teacher", required=True, help="teacher config YAML (see configs/teacher_template.yaml)")
    p.add_argument("--max-attempts", type=int, default=3)
    p.add_argument("--out")
    p.set_defaults(fn=cmd_teacher)

    p = sub.add_parser("labels", help="label-only SFT data (no teacher), the control baseline")
    _task_args(p, "train")
    p.add_argument("--out")
    p.set_defaults(fn=cmd_labels)

    p = sub.add_parser("sft", help="supervised fine-tuning from a YAML config")
    p.add_argument("config")
    p.add_argument("--train_file")
    p.add_argument("--output_dir")
    p.add_argument("--model")
    p.add_argument("--epochs", type=float)
    p.add_argument("--chat-template", help=CHAT_TEMPLATE_HELP)
    p.set_defaults(fn=cmd_sft)

    p = sub.add_parser("eval", help="generate with a student and score with the task")
    _task_args(p, "test")
    p.add_argument("--model", required=True)
    p.add_argument("--out")
    p.add_argument("--tag", default="")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--batch", type=int, default=0, help="prompts per batch; 0 = auto from free GPU memory")
    p.add_argument("--no-think", action="store_true", help="disable the chat template's thinking mode")
    p.add_argument("--chat-template", help=CHAT_TEMPLATE_HELP)
    p.set_defaults(fn=cmd_eval)

    p = sub.add_parser("grpo", help="GRPO with the task's score as reward")
    p.add_argument("config")
    _task_args(p, "train")
    p.add_argument("--exclude", help="JSONL whose meta.id rows are skipped (e.g. the SFT data)")
    p.add_argument("--filter-samples", type=int, default=0,
                   help="sample N completions per prompt first and keep prompts with 0 < pass rate < 1")
    p.add_argument("--filter-only", action="store_true", help="stop after the pass-rate filter")
    p.add_argument("--filter-gpu-memory", type=float, default=0.4, help="vLLM memory fraction for the filter")
    p.add_argument("--keep-ids", help="file of example ids to train on (pass_rate_kept.txt from --filter-only)")
    p.add_argument("--max-steps", type=int, help="override max_steps (e.g. a short smoke run)")
    p.add_argument("--epochs", type=float, help="set max_steps so each prompt is seen this many times")
    p.add_argument("--eval-every", type=int, help="validation check every N steps (0 = off); overrides the config")
    p.add_argument("--eval-examples", type=int, help="number of validation prompts; overrides the config")
    p.add_argument("--output_dir", help="override output_dir")
    p.add_argument("--model", help="override the starting model (any causal LM: hub id or local path)")
    p.add_argument("--chat-template", help=CHAT_TEMPLATE_HELP)
    p.set_defaults(fn=cmd_grpo)

    p = sub.add_parser("merge", help="merge a LoRA adapter into its base model")
    p.add_argument("--base", required=True)
    p.add_argument("--adapter", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_merge)

    p = sub.add_parser("doctor", help="environment report: GPU, precision, memory, disk, versions")
    p.set_defaults(fn=cmd_doctor)

    p = sub.add_parser("cost", help="teacher spend from the local response cache")
    p.add_argument("--teacher", required=True, help="teacher config YAML (its cache_path and price)")
    p.set_defaults(fn=cmd_cost)

    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
