"""One GRPO step on every architecture (transformers rollouts on CPU); an architecture whose image
inputs the installed TRL cannot pass through stops before training, with the reason."""

from pathlib import Path

import pytest
import yaml


def test_one_grpo_step_on_every_architecture(vlm, toy_data):
    from distillops_vision import compat
    from distillops_vision import model as vm
    from distillops_vision.cli import main

    cfg = {"model": vlm, "output_dir": "ckpt/g", "num_generations": 2, "batch_size": 2, "grad_accum": 1,
           "max_prompt_length": 512, "max_completion_length": 3, "max_steps": 1, "use_vllm": False,
           "eval_every": 0, "logging_steps": 1}
    Path("g.yaml").write_text(yaml.safe_dump(cfg))
    args = ["grpo", "g.yaml", "--task", "vision_toy:ColourTask", "--n", "2"]
    passed = compat.grpo_image_inputs()
    if "Mllama" in type(vm.load_model(vlm)).__name__ and passed is not None and "aspect_ratio_ids" not in passed:
        with pytest.raises(SystemExit, match="aspect_ratio_ids"):
            main(args)
        return
    main(args)
    assert (Path("ckpt/g") / "config.json").exists()


def test_grpo_with_lora_trains_the_language_model_only(qwen2vl, toy_data):
    import torch

    from distillops_vision import model as vm
    from distillops_vision.cli import main

    before = {k: v.clone() for k, v in vm.load_model(qwen2vl).state_dict().items()}
    cfg = {"model": qwen2vl, "output_dir": "ckpt/gl", "num_generations": 2, "batch_size": 2, "grad_accum": 1,
           "max_prompt_length": 512, "max_completion_length": 3, "max_steps": 1, "use_vllm": False, "lr": 1e-2,
           "eval_every": 0, "logging_steps": 1, "lora": True, "lora_r": 4, "lora_alpha": 8}
    Path("gl.yaml").write_text(yaml.safe_dump(cfg))
    main(["grpo", "gl.yaml", "--task", "vision_toy:ColourTask", "--n", "2"])
    saved = Path("ckpt/gl")
    assert (saved / "adapter_config.json").exists()  # a LoRA run saves the adapter (merge it with `merge`)
    main(["merge", "--base", qwen2vl, "--adapter", str(saved), "--out", "ckpt/gl-merged"])
    after = vm.load_model("ckpt/gl-merged").state_dict()
    # compared in bf16: on a GPU `merge` loads the base in bf16, so every weight is rounded to it
    changed = {k for k in before if not torch.equal(before[k].to(torch.bfloat16), after[k].to(torch.bfloat16))}
    assert not any(k.startswith("model.visual.") and ".merger." not in k for k in changed)  # encoder untouched
