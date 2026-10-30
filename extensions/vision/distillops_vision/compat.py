"""Which vision features the installed libraries support, and a report for `doctor`.

Vision SFT and evaluation need only transformers. GRPO with images needs a TRL whose GRPOTrainer
takes an `images` column (TRL >= 0.20) and, with vLLM rollouts, a vLLM that serves the model's
architecture with image input.
"""

from __future__ import annotations

from typing import Any

MIN_TRL_FOR_IMAGES = (0, 20)


def _version(mod: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(mod)
    except Exception:  # not installed
        return None


def _tuple(v: str | None) -> tuple[int, ...]:
    out = []
    for piece in (v or "0").split(".")[:2]:
        digits = "".join(c for c in piece if c.isdigit())
        out.append(int(digits or 0))
    return tuple(out)


def trl_supports_images() -> bool:
    return _tuple(_version("trl")) >= MIN_TRL_FOR_IMAGES


def vllm_supports(architectures: list[str]) -> bool | None:
    """Whether vLLM can serve one of `architectures` (None: vLLM not installed or not inspectable)."""
    try:
        from vllm import ModelRegistry

        supported = set(ModelRegistry.get_supported_archs())
    except Exception:
        return None
    return any(a in supported for a in architectures)


def require_grpo_support(use_vllm: bool, architectures: list[str]) -> None:
    """Stop with the fix before loading anything, instead of failing deep inside TRL or vLLM."""
    if not trl_supports_images():
        raise SystemExit(f"GRPO with images needs TRL >= {'.'.join(map(str, MIN_TRL_FOR_IMAGES))} "
                         f"(installed: {_version('trl')}); upgrade TRL")
    if use_vllm:
        ok = vllm_supports(architectures)
        if ok is False:
            raise SystemExit(f"the installed vLLM ({_version('vllm')}) does not serve {architectures}; set "
                             "use_vllm: false (rollouts with transformers, slower) or install a vLLM that does")


def grpo_image_inputs() -> set[str] | None:
    """Image inputs the installed TRL's GRPO passes to the model when it scores completions
    (None: this TRL does not say, so nothing is checked)."""
    import inspect

    try:
        from trl import GRPOTrainer
    except ImportError:
        return None
    fn = getattr(GRPOTrainer, "_get_per_token_logps_and_entropies", None)
    if fn is None:
        return None
    return {p for p in inspect.signature(fn).parameters if p not in ("self", "model", "input_ids", "attention_mask")}


def require_grpo_inputs(processor_keys: set[str], images_per_prompt: list[int]) -> None:
    """Stop before training on what TRL's GRPO cannot handle (the model would otherwise fail deep
    inside training):
    - image inputs it does not pass to the model (e.g. Mllama's aspect_ratio_ids with TRL 0.24);
    - pixel values with one row per image but no image_grid_thw to split them (LLaVA-style
      processors): TRL then treats them as one row per prompt, which holds only when every prompt
      has exactly one image."""
    if "pixel_values" in processor_keys and "image_grid_thw" not in processor_keys \
            and any(n != 1 for n in images_per_prompt):
        counts = sorted(set(images_per_prompt))
        raise SystemExit(f"this model's processor gives one row of pixel values per image, and TRL's GRPO splits "
                         f"them one row per prompt: GRPO needs exactly one image per prompt here (prompts have "
                         f"{counts} images). Vision SFT and evaluation are not affected.")
    passed = grpo_image_inputs()
    if passed is None:
        return
    dropped = sorted(processor_keys - passed - {"input_ids", "attention_mask", "token_type_ids"})
    if dropped:
        raise SystemExit(f"TRL {_version('trl')}'s GRPO does not pass {dropped} to the model, which this architecture "
                         "needs; vision SFT and evaluation work with it, GRPO does not. Use a TRL version that passes "
                         "these inputs, or an architecture whose processor makes only "
                         f"{sorted(passed)} besides the text.")


def vllm_ban(token_ids: list[int]) -> dict[str, Any]:
    """SamplingParams arguments that keep vLLM from sampling `token_ids` (a logit bias of -100),
    or {} with a warning when the installed vLLM has no logit_bias."""
    if not token_ids:
        return {}
    try:
        from vllm import SamplingParams
    except ImportError:
        return {}
    import inspect

    fields = set(getattr(SamplingParams, "__struct_fields__", ()) or ())
    if not fields:
        try:
            fields = set(inspect.signature(SamplingParams).parameters)
        except (TypeError, ValueError):
            fields = set()
    if "logit_bias" in fields:
        return {"logit_bias": {int(i): -100.0 for i in token_ids}}
    print("WARNING: this vLLM has no logit_bias: rollouts may sample image placeholder tokens, which breaks the "
          "training step that scores them", flush=True)
    return {}


def report(model: str | None = None, trust_remote_code: bool = False) -> dict[str, Any]:
    out: dict[str, Any] = {
        "versions": {m: _version(m) for m in ("transformers", "trl", "peft", "vllm", "pillow", "torchvision")},
        "grpo_with_images": trl_supports_images(),
    }
    if model:
        from .model import is_vision_config, load_config

        cfg = load_config(model, trust_remote_code)
        archs = list(getattr(cfg, "architectures", None) or [])
        out["model"] = {"name": model, "type": getattr(cfg, "model_type", None), "architectures": archs,
                        "takes_images": is_vision_config(cfg), "vllm_serves_it": vllm_supports(archs)}
    return out
