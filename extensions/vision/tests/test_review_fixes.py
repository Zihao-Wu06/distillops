"""Regression tests for the issues an independent review found (each test names its issue)."""

import sys
import types
from pathlib import Path

import pytest
import yaml

from distillops_vision import model as vm


def _grpo_yaml(model, out="ckpt/r", **kw):
    cfg = {"model": model, "output_dir": out, "num_generations": 2, "batch_size": 2, "grad_accum": 2,
           "max_prompt_length": 512, "max_completion_length": 3, "max_steps": 1, "use_vllm": False,
           "eval_every": 0, "logging_steps": 1, **kw}
    Path("r.yaml").write_text(yaml.safe_dump(cfg))
    return "r.yaml"


@pytest.mark.parametrize("task", ["vision_toy:MixedColourTask", "vision_toy:TwoImageTask"])
def test_grpo_refuses_llava_style_pixels_unless_one_image_per_prompt(llava, toy_data, task):
    # issue 1: TRL splits per-image pixel values one row per prompt; mixed or multi-image prompts crashed
    from distillops_vision.cli import main

    with pytest.raises(SystemExit, match="exactly one image per prompt"):
        main(["grpo", _grpo_yaml(llava), "--task", task])


@pytest.mark.parametrize("task", ["vision_toy:MixedColourTask", "vision_toy:TwoImageTask"])
def test_grpo_runs_mixed_and_multi_image_prompts_with_grid_based_pixels(qwen2vl, toy_data, task):
    from distillops_vision.cli import main

    main(["grpo", _grpo_yaml(qwen2vl), "--task", task])
    assert (Path("ckpt/r") / "config.json").exists()


def test_eval_never_batches_image_and_text_only_prompts_together(mllama, toy_data):
    # issue 2: Mllama's processor refuses such a batch; evaluation sorted them into one
    from distillops.cli import main

    main(["eval", "--task", "vision_toy:MixedColourTask", "--model", mllama, "--max-new-tokens", "3",
          "--batch", "4", "--tag", "m"])
    assert Path("results/mixed_colours/eval/m/metrics.json").exists()


def test_sft_on_mixed_rows_asks_for_batch_size_1_when_the_processor_needs_it(mllama, toy_data):
    from distillops.cli import main as distillops_main

    from distillops_vision.cli import main

    distillops_main(["labels", "--task", "vision_toy:MixedColourTask", "--out", "mixed.jsonl"])
    base = {"model": mllama, "train_file": "mixed.jsonl", "output_dir": "ckpt/m", "epochs": 1, "grad_accum": 1,
            "lora": False, "save_strategy": "no"}
    Path("m.yaml").write_text(yaml.safe_dump({**base, "batch_size": 2}))
    with pytest.raises(SystemExit, match="batch_size: 1"):
        main(["sft", "m.yaml"])
    Path("m.yaml").write_text(yaml.safe_dump({**base, "batch_size": 1}))
    main(["sft", "m.yaml"])


def test_trainable_parameters_next_to_lora_are_fp32(qwen2vl):
    # issue 3: PEFT left modules_to_save (the projector) in fp16/bf16
    import torch
    from peft import get_peft_model

    from distillops_vision.sft import VisionSFTConfig, lora_config

    lm = vm.load_model(qwen2vl, dtype=torch.bfloat16)
    peft_model = get_peft_model(lm, lora_config(lm, VisionSFTConfig(model="m", train_file="x", output_dir="y")))
    assert vm.upcast_trainable(peft_model) > 0
    assert all(p.dtype == torch.float32 for p in peft_model.parameters() if p.requires_grad)


def test_rollouts_never_sample_image_placeholders(monkeypatch, capsys):
    # issue 4: a sampled <|image_pad|> made the scoring forward fail
    from distillops_vision.grpo import rollout_kwargs

    hf = rollout_kwargs(False, [1, 2], 2, [7, 8])
    assert hf == {"eos_token_id": [1, 2], "pad_token_id": 2, "suppress_tokens": [7, 8]}

    class WithBias:
        __struct_fields__ = ("n", "logit_bias", "stop_token_ids")

    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(SamplingParams=WithBias))
    assert rollout_kwargs(True, [1], None, [7]) == {"stop_token_ids": [1], "logit_bias": {7: -100.0}}

    class NoBias:
        __struct_fields__ = ("n", "stop_token_ids")

    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(SamplingParams=NoBias))
    assert rollout_kwargs(True, [1], None, [7]) == {"stop_token_ids": [1]}
    assert "logit_bias" in capsys.readouterr().out


def test_encode_adds_no_special_tokens_of_its_own(qwen2vl):
    # issue 5: a tokenizer that adds BOS on top of a template that writes it gave <bos><bos>
    proc = vm.load_processor(qwen2vl)
    seen = {}
    real = proc.__class__.__call__

    def spy(self, *a, **kw):
        seen.update(kw)
        return real(self, *a, **kw)

    proc.__class__.__call__ = spy
    try:
        vm.encode(proc, ["hello"], [[]])
    finally:
        proc.__class__.__call__ = real
    assert seen["add_special_tokens"] is False


def test_usage_reads_a_vision_teacher_config(toy_data, capsys):
    # issue 6: distillops' usage rejected max_pixels / image_detail
    import asyncio

    from vision_toy import ColourTask

    from distillops_vision.cli import main
    from distillops_vision.teacher import VisionTeacher, VisionTeacherConfig
    from test_images_teacher import FakeClient

    cfg = {"base_url": "http://x", "model": "v", "cache_path": str(toy_data / "c.sqlite"), "max_pixels": 4096,
           "image_detail": "low"}
    Path("t.yaml").write_text(yaml.safe_dump(cfg))
    task = ColourTask()
    teacher = VisionTeacher(VisionTeacherConfig(**cfg), client=FakeClient())
    asyncio.run(teacher.complete(task.teacher_messages(task.load_examples("train")[0], 0)))
    main(["usage", "--teacher", "t.yaml"])
    assert "calls 1" in capsys.readouterr().out


def test_train_vision_with_lora_trains_the_encoder(qwen2vl):
    # issue 7: get_peft_model froze the encoder train_vision had left trainable
    from peft import get_peft_model

    from distillops_vision.sft import VisionSFTConfig, lora_config

    lm = vm.load_model(qwen2vl)
    vm.freeze_vision(lm, train_vision=True)
    cfg = VisionSFTConfig(model="m", train_file="x", output_dir="y", train_vision=True)
    peft_model = get_peft_model(lm, lora_config(lm, cfg))
    trainable = [n for n, p in peft_model.named_parameters() if p.requires_grad]
    # the encoder trains in full, as a module PEFT saves whole, next to LoRA on the language model
    assert any(".visual.modules_to_save." in n and ".blocks." in n for n in trainable)
    assert any("lora_" in n for n in trainable)


def test_image_size_comes_from_the_task_only():
    # issue 8: a max_pixels in the SFT / GRPO config resized images the teacher and evaluation never saw
    from dataclasses import fields

    from distillops_vision.grpo import VisionGRPOConfig
    from distillops_vision.sft import VisionSFTConfig

    assert "max_pixels" not in {f.name for f in fields(VisionSFTConfig)}
    assert "max_pixels" not in {f.name for f in fields(VisionGRPOConfig)}


def test_sft_eval_file_is_used_and_sftconfig_only_extras_are_explained(qwen2vl, toy_data, capsys):
    from distillops.cli import main as distillops_main

    from distillops_vision.cli import main

    distillops_main(["labels", "--task", "vision_toy:ColourTask", "--out", "rows.jsonl"])
    base = {"model": qwen2vl, "train_file": "rows.jsonl", "eval_file": "rows.jsonl", "output_dir": "ckpt/e",
            "epochs": 1, "batch_size": 2, "grad_accum": 1, "lora": False, "save_strategy": "no"}
    Path("e.yaml").write_text(yaml.safe_dump(base))
    main(["sft", "e.yaml"])
    assert "eval_loss" in capsys.readouterr().out
    Path("e.yaml").write_text(yaml.safe_dump({**base, "extra": {"packing": True}}))
    with pytest.raises(SystemExit, match="TrainingArguments"):
        main(["sft", "e.yaml"])


def test_fp32_projector_next_to_lora_runs_on_low_precision_inputs_without_autocast(qwen2vl):
    # on a GPU, TRL's GRPO generates without autocast: a fp32 projector got bf16 hidden states
    import torch
    from peft import get_peft_model

    from distillops_vision.sft import VisionSFTConfig, lora_config

    lm = vm.load_model(qwen2vl, dtype=torch.bfloat16)
    peft_model = get_peft_model(lm, lora_config(lm, VisionSFTConfig(model="m", train_file="x", output_dir="y")))
    vm.upcast_trainable(peft_model)
    proc = vm.load_processor(qwen2vl)
    from PIL import Image

    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "colour?"}]}]
    batch = proc(text=[vm.render(proc, msgs, add_generation_prompt=True)], images=[[Image.new("RGB", (56, 56))]],
                 return_tensors="pt")
    batch["pixel_values"] = batch["pixel_values"].to(torch.bfloat16)
    with torch.no_grad():
        out = peft_model.generate(**batch, max_new_tokens=2, do_sample=False)
    assert out.shape[1] == batch["input_ids"].shape[1] + 2
