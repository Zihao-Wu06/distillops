"""Vision GRPO on CPU (transformers rollouts), and the vLLM pass-rate filter's inputs (vLLM faked)."""

import json
import sys
import types
from pathlib import Path

import pytest
import yaml

from distillops_vision import model as vm


def test_rows_carry_the_rendered_prompt_the_images_and_the_id(vlm, toy_data):
    from vision_toy import ColourTask

    from distillops_vision.grpo import rows_and_reward

    task = ColourTask()
    examples = task.load_examples("train")
    proc = vm.load_processor(vlm)
    rows, reward = rows_and_reward(task, examples, proc, None)
    assert [r["ex_id"] for r in rows] == [task.example_id(ex) for ex in examples]
    assert rows[0]["prompt"].count(proc.image_token) == 1 and len(rows[0]["images"]) == 1
    w, h = rows[0]["images"][0].size
    assert w * h <= 1600  # sized by the part's max_pixels, as everywhere else
    good = [f"COLOUR: {ex['colour']}" for ex in examples]
    assert reward(good, [r["ex_id"] for r in rows]) == [1.0] * 4


def test_prompts_too_long_with_their_images_are_left_out(vlm, toy_data, capsys):
    from vision_toy import ColourTask

    from distillops_vision.grpo import fit_prompts

    task = ColourTask()
    examples = task.load_examples("train")
    proc = vm.load_processor(vlm)
    kept, stats = fit_prompts(task, examples, proc, 10_000, None)
    assert kept == examples and stats["too_long"] == 0
    kept, stats = fit_prompts(task, examples, proc, stats["longest"] - 1, None)
    assert stats["too_long"] >= 1 and len(kept) < len(examples) and "left out" in capsys.readouterr().out


def _grpo_cfg(model, out, **kw):
    cfg = {"model": model, "output_dir": out, "num_generations": 2, "batch_size": 2, "grad_accum": 1,
           "max_prompt_length": 512, "max_completion_length": 4, "max_steps": 2, "lr": 1e-3, "use_vllm": False,
           "logging_steps": 1, "save_steps": 100, "eval_every": 1, "eval_examples": 2, **kw}
    Path("grpo.yaml").write_text(yaml.safe_dump(cfg))
    return "grpo.yaml"


@pytest.fixture
def sft_model(qwen2vl, toy_data):
    """A tiny student fine-tuned on the toy task (so it carries trained_ids.txt)."""
    from distillops.cli import main as distillops_main

    from distillops_vision.cli import main

    distillops_main(["labels", "--task", "vision_toy:ColourTask", "--out", "data/sft/colours.jsonl"])
    Path("sft.yaml").write_text(yaml.safe_dump({"model": qwen2vl, "train_file": "data/sft/colours.jsonl",
                                                "output_dir": "ckpt/sft", "epochs": 1, "batch_size": 2,
                                                "grad_accum": 1, "lora": False, "save_strategy": "no"}))
    main(["sft", "sft.yaml"])
    return "ckpt/sft"


def test_vision_grpo_trains_validates_and_saves(sft_model, toy_data, capsys):
    from distillops.task import read_trained_ids

    from distillops_vision.cli import main

    main(["grpo", _grpo_cfg(sft_model, "ckpt/grpo"), "--task", "vision_toy:ColourTask"])
    log = capsys.readouterr().out
    assert "validation: 2 prompts (valid split)" in log and "weight change" in log
    result = json.loads(Path("ckpt/grpo/validation.json").read_text())
    assert result["steps"] == 2 and result["best_step"] is not None
    for d in (Path("ckpt/grpo"), Path("ckpt/grpo/last")):
        assert (d / "preprocessor_config.json").exists() or (d / "processor_config.json").exists()
        gen = json.loads((d / "generation_config.json").read_text())
        assert len(gen["eos_token_id"]) == 2  # EOS and <|im_end|>
    assert read_trained_ids("ckpt/grpo/last") == {f"colours:train:{i}" for i in range(4)}


def test_vision_grpo_without_validation_saves_the_final_model(sft_model, toy_data):
    from distillops_vision.cli import main

    main(["grpo", _grpo_cfg(sft_model, "ckpt/grpo2", eval_every=0, max_steps=1), "--task", "vision_toy:ColourTask"])
    assert (Path("ckpt/grpo2") / "config.json").exists() and not (Path("ckpt/grpo2") / "last").exists()


def test_pass_rate_filter_sends_images_to_vllm(qwen2vl, toy_data, monkeypatch):
    from vision_toy import ColourTask

    from distillops_vision import grpo

    seen = {}

    class LLM:
        def __init__(self, **kw):
            seen["init"] = kw

        def generate(self, inputs, params):
            seen["inputs"], seen["params"] = inputs, params
            outs = []
            for i, _ in enumerate(inputs):  # first prompt always right, the others mixed
                texts = ["COLOUR: red"] * params.n if i == 0 else ["COLOUR: red", "COLOUR: blue"]
                outs.append(types.SimpleNamespace(outputs=[types.SimpleNamespace(text=t) for t in texts]))
            return outs

    class SamplingParams:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(LLM=LLM, SamplingParams=SamplingParams))
    monkeypatch.setattr("distillops.memory.require_free_memory", lambda *a, **k: None)
    task = ColourTask()
    examples = task.load_examples("train")
    proc = vm.load_processor(qwen2vl)
    scores, stats = grpo.estimate_pass_rates(qwen2vl, task, examples, proc, n=2, max_tokens=4, max_prompt_length=512)
    assert seen["init"]["limit_mm_per_prompt"] == {"image": 1} and seen["init"]["max_model_len"] == 516
    first = seen["inputs"][0]
    assert proc.image_token in first["prompt"] and len(first["multi_modal_data"]["image"]) == 1
    assert seen["params"].stop_token_ids == [proc.tokenizer.convert_tokens_to_ids("<|im_end|>")]
    assert stats["mixed"] >= 1 and len(scores) == 4


def test_grpo_stops_early_when_trl_cannot_take_images(monkeypatch):
    from distillops_vision import compat

    monkeypatch.setattr(compat, "_version", lambda m: "0.19.0" if m == "trl" else None)
    with pytest.raises(SystemExit, match="TRL >= 0.20"):
        compat.require_grpo_support(False, ["Qwen2VLForConditionalGeneration"])
