"""python -m distillops_vision <command>: the vision steps of the distillops workflow.

  teacher    teacher answers on prompts with images -> verify -> accepted SFT rows (resumable)
  sft        SFT of a vision-language student
  grpo       GRPO with Task.score as reward (and the pass-rate filter)
  usage      teacher calls and tokens from the response cache (takes a vision teacher config)
  merge      merge a LoRA adapter into its vision-language base model
  doctor     environment report: distillops' own, plus the vision model and library versions
  check-task a task's prompts, image placeholders and loss mask on one row

`python -m distillops labels` and `eval` work on a VisionTask unchanged.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from distillops.cli import (_exclude, _task_args, args_suffix, load_task, subsample, task_fingerprint, task_spec,
                            template_args)


def cmd_teacher(a: argparse.Namespace) -> None:
    from distillops.rejection import check_run_settings, rejection_sample

    from .teacher import VisionTeacher, VisionTeacherConfig, image_fingerprint

    tcfg = VisionTeacherConfig.from_yaml(a.teacher)
    spec, kwargs = task_spec(a.task, a.task_arg)
    task = load_task(a.task, a.task_arg)
    split = task.load_examples(a.split)
    examples = subsample(split, a.n, a.seed)
    out = Path(a.out or f"results/{task.name}/teacher_{a.split}_n{a.n}{args_suffix(kwargs)}")
    override = tcfg.max_pixels if tcfg.max_pixels is not None else getattr(task, "teacher_max_pixels", None)
    # a rerun resumes this folder: same teacher, prompts, images, code and task settings
    check_run_settings(out, {"task": spec, "task_args": kwargs, "split": a.split,
                             "teacher": {**tcfg.identity(), "max_pixels": override},
                             **task_fingerprint(task, split), "teacher_images": image_fingerprint(task, split)})
    teacher = VisionTeacher(tcfg, max_pixels=override)
    stats = asyncio.run(rejection_sample(task, teacher, examples, out, a.max_attempts))
    print(stats)
    print(f"SFT data: {out / 'accepted.jsonl'}")


def cmd_sft(a: argparse.Namespace) -> None:
    from distillops.io import write_json

    from .sft import VisionSFTConfig, train

    cfg = VisionSFTConfig.from_yaml(a.config, train_file=a.train_file, output_dir=a.output_dir, model=a.model,
                                    epochs=a.epochs, chat_template=a.chat_template)
    res = train(cfg)
    write_json(Path(cfg.output_dir) / "sft_summary.json", res)
    print(res)


def cmd_grpo(a: argparse.Namespace) -> None:
    from distillops.grpo import diagnose_pass_rates, filter_by_signal, split_validation, steps_for_epochs
    from distillops.io import write_json
    from distillops.task import check_ids, read_trained_ids

    from .grpo import VisionGRPOConfig, estimate_pass_rates, fit_prompts, make_evaluator, rows_and_reward, train
    from .model import load_processor

    cfg = VisionGRPOConfig.from_yaml(a.config, model=a.model, chat_template=a.chat_template)
    cfg.chat_template_kwargs = {**(cfg.chat_template_kwargs or {}), **template_args(a.template_arg)}
    for field in ("max_steps", "output_dir", "eval_every", "eval_examples"):
        value = getattr(a, field)
        if value is not None and (field != "max_steps" or value):
            setattr(cfg, field, value)
    task = load_task(a.task, a.task_arg)
    processor = load_processor(cfg.model, cfg.chat_template, cfg.trust_remote_code)
    tk = cfg.chat_template_kwargs
    split = task.load_examples(a.split)
    check_ids(task, split)
    examples = subsample(_exclude(task, split, a.exclude), a.n, a.seed)
    examples, _ = fit_prompts(task, examples, processor, cfg.max_prompt_length, None, tk)
    valid: list = []
    if cfg.eval_every > 0 and cfg.eval_examples > 0 and not a.filter_only:
        trained = read_trained_ids(cfg.model)
        if trained is None and Path(cfg.model).is_dir():
            print(f"WARNING: {cfg.model} has no record of the examples it was trained on (trained_ids.txt), so "
                  "validation may contain prompts it learned and pick the wrong checkpoint. Pass --exclude <its SFT "
                  "data .jsonl>, or give the task a 'valid' split SFT did not use.", flush=True)
        examples, valid, source = split_validation(task, examples, cfg.eval_examples, a.seed, exclude_ids=trained)
        if valid:
            valid, _ = fit_prompts(task, valid, processor, cfg.max_prompt_length, None, tk)
            print(f"validation: {len(valid)} prompts ({source}), checked every {cfg.eval_every} steps", flush=True)
        else:
            print(f"WARNING: no validation ({source}), so checkpoint selection is off and output_dir will hold the "
                  "final model. For selection, give the task a 'valid' split that SFT did not use.", flush=True)
    if a.keep_ids:
        keep = {line.strip() for line in Path(a.keep_ids).read_text(encoding="utf-8").splitlines() if line.strip()}
        examples = [ex for ex in examples if task.example_id(ex) in keep]
        print(f"{len(examples)} prompts from {a.keep_ids}", flush=True)
    if a.filter_samples:
        scores, stats = estimate_pass_rates(cfg.model, task, examples, processor, n=a.filter_samples,
                                            temperature=cfg.temperature, max_tokens=cfg.max_completion_length,
                                            template_kwargs=tk, gpu_memory_utilization=a.filter_gpu_memory,
                                            max_prompt_length=cfg.max_prompt_length, max_pixels=None,
                                            trust_remote_code=cfg.trust_remote_code)
        examples = filter_by_signal(examples, scores)
        stats["advice"] = diagnose_pass_rates(stats)
        print("pass-rate filter:", json.dumps(stats), f"-> {len(examples)} prompts kept", flush=True)
        for note in stats["advice"]:
            print("  *", note, flush=True)
        write_json(Path(cfg.output_dir) / "pass_rate_filter.json", stats)
        (Path(cfg.output_dir) / "pass_rate_kept.txt").write_text(
            "\n".join(task.example_id(ex) for ex in examples), encoding="utf-8")
        if a.filter_only:  # train in a fresh process: vLLM may not release all GPU memory
            return
    if a.epochs:
        cfg.max_steps = steps_for_epochs(len(examples), a.epochs, cfg)
        print(f"{len(examples)} prompts x {a.epochs} epochs -> {cfg.max_steps} steps", flush=True)
    rows, reward = rows_and_reward(task, examples, processor, None, tk)
    evaluator = make_evaluator(task, valid, cfg.max_completion_length, None, tk) if valid else None
    print(train(cfg, [reward], rows=rows, evaluator=evaluator, processor=processor))


def cmd_usage(a: argparse.Namespace) -> None:
    """distillops' `usage` for a vision teacher config (it holds fields distillops' config does not)."""
    import tempfile
    from dataclasses import fields

    import yaml

    from distillops.cli import cmd_usage as distillops_usage
    from distillops.teacher import TeacherConfig

    raw = yaml.safe_load(Path(a.teacher).read_text(encoding="utf-8"))
    keep = {f.name for f in fields(TeacherConfig)}
    with tempfile.TemporaryDirectory() as d:
        base = Path(d) / "teacher.yaml"
        base.write_text(yaml.safe_dump({k: v for k, v in raw.items() if k in keep}), encoding="utf-8")
        distillops_usage(argparse.Namespace(teacher=str(base)))


def cmd_merge(a: argparse.Namespace) -> None:
    """Merge a LoRA adapter into its vision-language base model."""
    from peft import PeftModel

    from distillops.chat import save_stop_tokens
    from distillops.precision import compute_dtype
    from distillops.task import read_trained_ids, write_trained_ids

    from .model import TemplateView, load_model, load_processor

    base = load_model(a.base, dtype=compute_dtype(), trust_remote_code=a.trust_remote_code)
    merged = PeftModel.from_pretrained(base, a.adapter).merge_and_unload()
    merged.save_pretrained(a.out)
    src = a.adapter if (Path(a.adapter) / "preprocessor_config.json").exists() else a.base
    processor = load_processor(src, trust_remote_code=a.trust_remote_code)
    processor.save_pretrained(a.out)
    save_stop_tokens(a.out, TemplateView(processor))
    ids = read_trained_ids(a.adapter)
    if ids is not None:
        write_trained_ids(a.out, ids)
    print(f"merged {a.adapter} into {a.base} -> {a.out}")


def cmd_doctor(a: argparse.Namespace) -> None:
    from distillops.preflight import doctor

    from .compat import report

    doctor()  # distillops' own report (GPU, precision, disk, versions)
    rep = report(a.model, a.trust_remote_code)
    for k, v in rep["versions"].items():
        print(f"{k:16s} {v or '-'}")
    print(f"{'grpo with images':16s} {'yes' if rep['grpo_with_images'] else 'no (TRL >= 0.20 needed)'}")
    for k, v in (rep.get("model") or {}).items():
        print(f"{'model ' + k:16s} {v}")


def cmd_check_task(a: argparse.Namespace) -> None:
    from . import images
    from .model import (lengths, load_model, load_processor, placeholder_ids, render, suppressed_ids, tokenizer_of,
                        vision_modules)

    task = load_task(a.task, a.task_arg)
    examples = subsample(task.load_examples(a.split), a.n or 1, a.seed)
    processor = load_processor(a.model, a.chat_template, a.trust_remote_code)
    ex = examples[0]
    print(f"{len(examples)} examples; id {task.example_id(ex)!r}")
    print("teacher prompt:", json.dumps(task.teacher_messages(ex, 0), ensure_ascii=False)[:1500])
    msgs = task.student_messages(ex)
    print("student prompt as rendered:\n" + render(processor, msgs, add_generation_prompt=True))
    print("images:", [(p["path"], images.pixels(p)) for p in images.image_parts(msgs)])
    try:
        row = task.gold_sft_example(ex)
    except NotImplementedError:
        row = None
    if row:
        p_len, f_len, aligned = lengths(processor, row, None)
        print(f"training row (gold): {f_len} tokens, of which {p_len} prompt (masked) and {f_len - p_len} answer "
              f"(loss){'' if aligned else ' -- WARNING: prompt is not a prefix of the full row'}")
        print("answer (loss):", render(processor, row["prompt"] + row["completion"])[
            len(render(processor, row["prompt"], add_generation_prompt=True)):])
    lm = load_model(a.model, trust_remote_code=a.trust_remote_code)
    tok = tokenizer_of(processor)
    print("vision modules:", vision_modules(lm) or "none (e.g. an encoder-free model)")
    print("image placeholder tokens:", tok.convert_ids_to_tokens(sorted(placeholder_ids(lm, processor))))
    print("never generated:", len(suppressed_ids(lm, processor)), "token ids")


def _vision_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--trust-remote-code", action="store_true", help="run the model repository's own code")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m distillops_vision", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("teacher", help="teacher answers on image prompts -> verify -> accepted SFT rows")
    _task_args(p, "train")
    p.add_argument("--teacher", required=True, help="teacher config YAML (OpenAI-compatible, with image input)")
    p.add_argument("--max-attempts", type=int, default=3)
    p.add_argument("--out")
    p.set_defaults(fn=cmd_teacher)

    p = sub.add_parser("sft", help="SFT of a vision-language student from a YAML config")
    p.add_argument("config")
    p.add_argument("--train_file")
    p.add_argument("--output_dir")
    p.add_argument("--model")
    p.add_argument("--epochs", type=float)
    p.add_argument("--chat-template", help="a Jinja file or string; default: the model's own")
    p.set_defaults(fn=cmd_sft)

    p = sub.add_parser("grpo", help="GRPO on image prompts with Task.score as reward")
    p.add_argument("config")
    _task_args(p, "train")
    p.add_argument("--exclude", help="JSONL whose meta.id rows are skipped (e.g. the SFT data)")
    p.add_argument("--filter-samples", type=int, default=0,
                   help="sample N completions per prompt first and keep prompts whose scores differ")
    p.add_argument("--filter-only", action="store_true", help="stop after the pass-rate filter")
    p.add_argument("--filter-gpu-memory", type=float, default=0.4, help="vLLM memory fraction for the filter")
    p.add_argument("--keep-ids", help="file of example ids to train on (pass_rate_kept.txt from --filter-only)")
    p.add_argument("--max-steps", type=int, help="override max_steps (e.g. a short smoke run)")
    p.add_argument("--epochs", type=float, help="set max_steps so each prompt is seen this many times")
    p.add_argument("--eval-every", type=int, help="validation check every N steps (0 = off); overrides the config")
    p.add_argument("--eval-examples", type=int, help="number of validation prompts; overrides the config")
    p.add_argument("--output_dir", help="override output_dir")
    p.add_argument("--model", help="override the starting model")
    p.add_argument("--chat-template", help="a Jinja file or string; default: the model's own")
    p.add_argument("--template-arg", action="append", default=[], metavar="KEY=VALUE",
                   help="option for the chat template (e.g. enable_thinking=false)")
    p.set_defaults(fn=cmd_grpo)

    p = sub.add_parser("usage", help="teacher calls and tokens, from the response cache (vision teacher config)")
    p.add_argument("--teacher", required=True)
    p.set_defaults(fn=cmd_usage)

    p = sub.add_parser("merge", help="merge a LoRA adapter into its vision-language base model")
    p.add_argument("--base", required=True)
    p.add_argument("--adapter", required=True)
    p.add_argument("--out", required=True)
    _vision_model_args(p)
    p.set_defaults(fn=cmd_merge)

    p = sub.add_parser("doctor", help="environment report, with the vision model and library versions")
    p.add_argument("--model", help="a vision-language model to inspect (optional)")
    _vision_model_args(p)
    p.set_defaults(fn=cmd_doctor)

    p = sub.add_parser("check-task", help="a task's prompts, image placeholders and loss mask on one example")
    _task_args(p, "train")
    p.add_argument("--model", required=True, help="the student (its processor renders the prompts)")
    p.add_argument("--chat-template", help="a Jinja file or string; default: the model's own")
    _vision_model_args(p)
    p.set_defaults(fn=cmd_check_task)

    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
