"""What the extension uses from distillops (so a change there fails here first), and the small commands."""

import inspect
import json
from pathlib import Path

import pytest
import yaml

# every distillops name the extension imports, with the parameters it relies on
CONTRACT = {
    "distillops.task": {"Task": None, "Verdict": None, "check_ids": ["task", "examples"],
                        "read_trained_ids": ["model_dir"], "write_trained_ids": ["model_dir", "ids"]},
    "distillops.teacher": {"Teacher._key": ["self", "messages", "sample_id", "params", "with_endpoint"],
                           "Teacher._call_with_retry": ["self", "messages", "params", "extra"],
                           "TeacherConfig": None},
    "distillops.rejection": {"rejection_sample": ["task", "teacher", "examples", "out_dir", "max_attempts"],
                             "check_run_settings": ["out_dir", "settings"]},
    "distillops.cli": {"task_spec": ["spec", "task_args"], "load_task": ["spec", "task_args"],
                       "subsample": ["examples", "n", "seed"], "args_suffix": ["kwargs"],
                       "task_fingerprint": ["task", "examples"], "template_args": ["pairs", "no_think"],
                       "_task_args": ["p", "split"], "_exclude": ["task", "examples", "path"], "main": ["argv"],
                       "cmd_usage": ["a"]},
    "distillops.evaluate": {"evaluate_task": ["task", "model", "examples", "out_dir", "generate_fn"]},
    "distillops.sft": {"SFTRunConfig": None, "SFT_FIELDS": None,
                       "sft_disk_bytes": ["params", "lora", "args", "n_rows", "merged_bytes_per_param"]},
    "distillops.grpo": {"GRPORunConfig": None, "GRPO_FIELDS": None,
                        "task_rows_and_reward": ["task", "examples", "render"], "signal_stats": ["scores"],
                        "filter_by_signal": ["rows", "scores"], "diagnose_pass_rates": ["stats"],
                        "split_validation": ["task", "examples", "n", "seed", "exclude_ids"],
                        "validation_selector": ["evaluate", "every", "patience", "save_best"],
                        "mean_and_se": ["scores"], "entropy_guard": ["floor_ratio", "patience"],
                        "steps_for_epochs": ["n_prompts", "epochs", "cfg"], "fit_micro_batch": ["cfg"],
                        "grpo_disk_bytes": ["params", "lora", "args", "validation", "bytes_per_param"],
                        "_snapshot": ["model"], "_relative_change": ["before", "model"]},
    "distillops.chat": {"end_of_turn_id": ["tok"], "stop_token_ids": ["tok", "model_eos"],
                        "save_stop_tokens": ["model_dir", "tok"], "untrained_token_ids": ["model", "tok", "token_ids",
                                                                                           "written_ids"],
                        "resolve_template": ["chat_template"], "device": []},
    "distillops.memory": {"adaptive_chunks": ["n", "batch", "run", "limit", "log"], "max_sequences": ["model", "seq_len"],
                          "require_free_memory": ["fraction", "setting"]},
    "distillops.precision": {"train_settings": ["full_finetune"], "compute_dtype": [], "describe": [],
                             "dtype_name": ["dtype"]},
    "distillops.preflight": {"check_disk": ["out_dir", "need_bytes", "what"], "param_count": ["model"],
                             "stored_bytes": ["model"], "min_expected_change": ["lr", "steps", "weight_rms"],
                             "doctor": []},
    "distillops.trainargs": {"merge_extra": ["args", "extra", "fields", "what"]},
    "distillops.io": {"write_json": None, "read_jsonl": None},
}


@pytest.mark.parametrize("module", sorted(CONTRACT))
def test_distillops_still_offers_what_the_extension_uses(module):
    import importlib

    mod = importlib.import_module(module)
    for name, params in CONTRACT[module].items():
        obj = mod
        for part in name.split("."):
            assert hasattr(obj, part), f"{module}.{name} is gone: distillops_vision uses it"
            obj = getattr(obj, part)
        if params is not None:
            have = list(inspect.signature(obj).parameters)
            missing = [p for p in params if p not in have]
            assert not missing, f"{module}.{name} lost parameters {missing}: distillops_vision passes them"


def test_vision_configs_accept_every_distillops_field():
    from dataclasses import fields

    from distillops.grpo import GRPORunConfig
    from distillops.sft import SFTRunConfig

    from distillops_vision.grpo import VisionGRPOConfig
    from distillops_vision.sft import VisionSFTConfig

    assert {f.name for f in fields(SFTRunConfig)} <= {f.name for f in fields(VisionSFTConfig)}
    assert {f.name for f in fields(GRPORunConfig)} <= {f.name for f in fields(VisionGRPOConfig)}


def test_check_task_and_doctor_commands(qwen2vl, toy_data, capsys):
    from distillops_vision.cli import main

    main(["check-task", "--task", "vision_toy:ColourTask", "--model", qwen2vl])
    out = capsys.readouterr().out
    assert "<|image_pad|>" in out and "COLOUR: red" in out and "model.visual" in out
    main(["doctor", "--model", qwen2vl])
    out = capsys.readouterr().out
    assert "precision" in out and "model takes_images True" in " ".join(out.split())


def test_merge_command_combines_a_lora_adapter_with_its_vision_base(qwen2vl, toy_data):
    from distillops.cli import main as distillops_main
    from distillops.task import read_trained_ids

    from distillops_vision import model as vm
    from distillops_vision.cli import main

    distillops_main(["labels", "--task", "vision_toy:ColourTask", "--out", "rows.jsonl"])
    Path("l.yaml").write_text(yaml.safe_dump({"model": qwen2vl, "train_file": "rows.jsonl", "output_dir": "lora",
                                              "epochs": 1, "batch_size": 2, "grad_accum": 1, "lora": True,
                                              "lora_r": 4, "lora_alpha": 8, "save_strategy": "no"}))
    main(["sft", "l.yaml"])
    main(["merge", "--base", qwen2vl, "--adapter", "lora", "--out", "merged-again"])
    assert vm.is_vision("merged-again") and read_trained_ids("merged-again") == read_trained_ids("lora/merged")
    gen = json.loads(Path("merged-again/generation_config.json").read_text())
    assert len(gen["eos_token_id"]) == 2
