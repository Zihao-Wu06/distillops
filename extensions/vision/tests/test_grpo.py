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


def test_grpo_stops_early_on_a_trl_without_the_apis_it_uses(monkeypatch):
    from distillops_vision import compat

    monkeypatch.setattr(compat, "_version", lambda m: "1.17.1" if m == "trl" else None)
    with pytest.raises(SystemExit, match=r"trl>=0.20,<0.28"):
        compat.require_grpo_support(False, ["Qwen2VLForConditionalGeneration"])


def test_grpo_dataset_keeps_images_decoded_in_memory(vlm, toy_data):
    from vision_toy import ColourTask

    from distillops_vision.grpo import image_dataset, rows_and_reward

    task = ColourTask()
    rows, _ = rows_and_reward(task, task.load_examples("train"), vm.load_processor(vlm), None)
    ds = image_dataset(rows)
    assert len(ds) == len(rows)
    first = ds[0]
    assert first["images"][0] is rows[0]["images"][0]  # the same object: no PNG round trip per read
    assert first["prompt"] == rows[0]["prompt"] and first["ex_id"] == rows[0]["ex_id"]
    assert ds[0]["images"][0] is ds[0]["images"][0]
    plain = image_dataset([{"prompt": "p", "ex_id": "a"}])
    assert plain[0] == {"prompt": "p", "ex_id": "a"}


def test_validation_with_vllm_syncs_once_then_scores_greedy_answers(vlm, toy_data, monkeypatch):
    from vision_toy import ColourTask

    from distillops_vision.grpo import make_evaluator

    calls = {"sync": 0}

    class SamplingParams:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    class LLM:
        def generate(self, inputs, sampling_params, use_tqdm):
            calls["inputs"], calls["params"] = inputs, sampling_params
            return [types.SimpleNamespace(outputs=[types.SimpleNamespace(text="COLOUR: red")]) for _ in inputs]

    class Trainer:
        processing_class = vm.load_processor(vlm)
        state = types.SimpleNamespace(global_step=3)
        _last_loaded_step = 2
        llm = LLM()

        def _move_model_to_vllm(self):
            calls["sync"] += 1

    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(SamplingParams=SamplingParams))
    task = ColourTask()
    examples = task.load_examples("train")
    evaluate = make_evaluator(task, examples, 8, None)
    trainer = Trainer()
    mean, _ = evaluate.with_vllm(trainer, {"stop_token_ids": [1]})
    assert calls["sync"] == 1 and trainer._last_loaded_step == 3  # the next rollout skips its sync
    assert calls["params"].temperature == 0.0 and calls["params"].max_tokens == 8
    assert calls["params"].stop_token_ids == [1]
    assert len(calls["inputs"]) == 4 and len(calls["inputs"][0]["multi_modal_data"]["image"]) == 1
    assert mean == sum(task.score(ex, "COLOUR: red") for ex in examples) / 4
    evaluate.with_vllm(trainer, {})
    assert calls["sync"] == 1  # weights already in vLLM for this step


def test_cpu_threads_is_at_least_one():
    from distillops_vision.grpo import cpu_threads

    assert cpu_threads() >= 1


def _fake_trainer(llm_model, peft=False):
    import torch

    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 2))
    model[0].weight.requires_grad_(False)  # a frozen module (e.g. the image encoder)
    model[0].bias.requires_grad_(False)
    if peft:
        model.peft_config = {}
    engine = types.SimpleNamespace(model_executor=types.SimpleNamespace(
        driver_worker=types.SimpleNamespace(model_runner=types.SimpleNamespace(model=llm_model))))
    resets = []
    return types.SimpleNamespace(
        vllm_mode="colocate", is_fsdp_enabled=False, model=model,
        accelerator=types.SimpleNamespace(state=types.SimpleNamespace(deepspeed_plugin=None)),
        llm=types.SimpleNamespace(llm_engine=engine, reset_prefix_cache=lambda: resets.append(1)),
        _fix_param_name_to_vllm=lambda n: "vllm." + n, resets=resets)


def test_weight_sync_sends_trained_parameters_in_one_call(monkeypatch):
    from trl import GRPOTrainer

    from distillops_vision.grpo import trainer_class

    calls = []
    llm_model = types.SimpleNamespace(load_weights=lambda w: calls.append([n for n, _ in w]))
    monkeypatch.setattr(GRPOTrainer, "_move_model_to_vllm", lambda self: calls.append("trl"))
    t = _fake_trainer(llm_model)
    trainer_class()._move_model_to_vllm(t)
    assert calls == [["vllm.1.weight", "vllm.1.bias"]] and t.resets == [1]  # frozen ones not sent


def test_weight_sync_falls_back_to_trl(monkeypatch, capsys):
    from trl import GRPOTrainer

    from distillops_vision.grpo import trainer_class

    calls = []
    monkeypatch.setattr(GRPOTrainer, "_move_model_to_vllm", lambda self: calls.append("trl"))
    sync = trainer_class()._move_model_to_vllm
    sync(_fake_trainer(types.SimpleNamespace(load_weights=lambda w: calls.append("fast")), peft=True))  # LoRA
    old = _fake_trainer(types.SimpleNamespace())  # a vLLM model without load_weights (another version)
    sync(old)
    sync(old)
    assert calls == ["trl"] * 3
    assert capsys.readouterr().out.count("one-call weight sync") == 1  # said once
