"""Generation, evaluation through distillops' own `eval` command, and vision SFT, on both architectures."""

import json
from pathlib import Path

import pytest
import yaml

from distillops_vision import model as vm


def test_generate_returns_one_completion_per_prompt_in_order(vlm, toy_data):
    from vision_toy import ColourTask

    from distillops_vision.generate import generate_vision

    task = ColourTask()
    msgs = [task.student_messages(ex) for ex in task.load_examples("test")]
    a = generate_vision(vlm, msgs, max_new_tokens=4, batch_size=2)
    b = generate_vision(vlm, msgs[::-1], max_new_tokens=4, batch_size=1)
    assert len(a) == 3 and a == b[::-1]  # greedy, so batching and order do not change the outputs


def test_distillops_eval_command_runs_a_vision_task(vlm, toy_data):
    from distillops.cli import main

    main(["eval", "--task", "vision_toy:ColourTask", "--model", vlm, "--max-new-tokens", "4", "--tag", "tiny"])
    metrics = json.loads(Path("results/colours/eval/tiny/metrics.json").read_text())
    assert metrics["n"] == 3 and metrics["max_pixels"] == 1600 and 0 <= metrics["score"] <= 1
    preds = [json.loads(l) for l in Path("results/colours/eval/tiny/predictions.jsonl").read_text().splitlines()]
    assert [p["id"] for p in preds] == [f"colours:test:{i}" for i in range(3)]


def _rows(toy_data):
    from distillops.cli import main

    main(["labels", "--task", "vision_toy:ColourTask", "--out", "data/sft/colours.jsonl"])  # distillops' command
    return "data/sft/colours.jsonl"


def test_collator_puts_loss_on_the_answer_only(vlm, toy_data):
    from distillops_vision.sft import Collator, length_filter

    proc = vm.load_processor(vlm)
    rows = [json.loads(l) for l in Path(_rows(toy_data)).read_text().splitlines()][:2]
    kept, stats = length_filter(rows, proc, 1024, None)
    assert stats["rows_kept"] == 2 and stats["prompt_boundary_mismatch"] == 0
    batch = Collator(proc, None)(kept)
    labels = batch["labels"]
    for i, r in enumerate(kept):
        answer = labels[i][labels[i] != -100]
        text = proc.tokenizer.decode(answer)
        assert text.startswith("COLOUR: ") and r["completion"][0]["content"] in text  # the answer and end of turn
        assert (labels[i, : r["_prompt_len"]] == -100).all()  # prompt and image tokens carry no loss
    assert "pixel_values" in batch


def test_length_filter_drops_rows_longer_than_max_length_with_their_images(vlm, toy_data):
    from distillops_vision.sft import length_filter

    proc = vm.load_processor(vlm)
    rows = [json.loads(l) for l in Path(_rows(toy_data)).read_text().splitlines()]
    longest = length_filter(rows, proc, 10_000, None)[1]["len_max"]
    kept, stats = length_filter(rows, proc, longest - 1, None)
    assert stats["dropped_too_long"] >= 1 and len(kept) == len(rows) - stats["dropped_too_long"]


@pytest.mark.parametrize("lora", [False, True])
def test_vision_sft_trains_and_saves_a_usable_model(qwen2vl, toy_data, lora):
    from distillops.cli import main as distillops_main
    from distillops.task import read_trained_ids

    from distillops_vision.cli import main

    train_file = _rows(toy_data)
    cfg = {"model": qwen2vl, "train_file": train_file, "output_dir": "ckpt/sft", "epochs": 1, "batch_size": 2,
           "grad_accum": 1, "lr": 1e-3, "lora": lora, "lora_r": 4, "lora_alpha": 8, "logging_steps": 1,
           "save_strategy": "no"}
    Path("sft.yaml").write_text(yaml.safe_dump(cfg))
    main(["sft", "sft.yaml"])
    out = Path("ckpt/sft") / ("merged" if lora else "")
    summary = json.loads(Path("ckpt/sft/sft_summary.json").read_text())
    assert summary["length_filter"]["rows_kept"] == 4 and summary["vision"]["vision_modules"] == ["model.visual"]
    assert read_trained_ids(out) == {f"colours:train:{i}" for i in range(4)}
    gen = json.loads((out / "generation_config.json").read_text())
    proc = vm.load_processor(str(out))
    assert proc.tokenizer.convert_tokens_to_ids("<|im_end|>") in gen["eos_token_id"]  # stops at the end of turn
    distillops_main(["eval", "--task", "vision_toy:ColourTask", "--model", str(out), "--max-new-tokens", "4"])


def test_full_fine_tuning_runs_on_every_architecture(vlm, toy_data):
    from distillops.cli import main as distillops_main

    from distillops_vision.cli import main

    Path("a.yaml").write_text(yaml.safe_dump({"model": vlm, "train_file": _rows(toy_data), "output_dir": "ckpt/a",
                                              "epochs": 1, "batch_size": 2, "grad_accum": 1, "lora": False,
                                              "save_strategy": "no"}))
    main(["sft", "a.yaml"])
    assert vm.is_vision("ckpt/a")
    distillops_main(["eval", "--task", "vision_toy:ColourTask", "--model", "ckpt/a", "--max-new-tokens", "3"])


def test_lora_adapts_the_language_model_only(qwen2vl):
    from peft import get_peft_model

    from distillops_vision.sft import VisionSFTConfig, lora_config

    lm = vm.load_model(qwen2vl)
    cfg = VisionSFTConfig(model=qwen2vl, train_file="x", output_dir="y", lora_r=4, lora_alpha=8)
    peft_model = get_peft_model(lm, lora_config(lm, cfg))
    lora_paths = [n for n, _ in peft_model.named_parameters() if "lora_" in n]
    assert lora_paths and not any(".visual." in n for n in lora_paths)
    trainable = [n for n, p in peft_model.named_parameters() if p.requires_grad and "lora_" not in n]
    assert trainable and all(".merger." in n for n in trainable)  # the projector trains fully


def test_frozen_vision_encoder_does_not_change_in_sft(qwen2vl, toy_data):
    import torch

    from distillops_vision.cli import main

    before = {k: v.clone() for k, v in vm.load_model(qwen2vl).state_dict().items()}
    cfg = {"model": qwen2vl, "train_file": _rows(toy_data), "output_dir": "ckpt/full", "epochs": 1, "batch_size": 2,
           "grad_accum": 1, "lr": 1e-2, "lora": False, "save_strategy": "no", "logging_steps": 1}
    Path("full.yaml").write_text(yaml.safe_dump(cfg))
    main(["sft", "full.yaml"])
    after = vm.load_model("ckpt/full").state_dict()
    changed = {k for k in before if not torch.equal(before[k].float(), after[k].float())}
    assert changed and not any(k.startswith("model.visual.") and ".merger." not in k for k in changed)
    assert any(".merger." in k for k in changed) and any("language_model" in k for k in changed)
