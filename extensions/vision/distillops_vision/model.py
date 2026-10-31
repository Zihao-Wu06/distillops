"""Vision-language students: loading, prompt rendering, token counts and the vision modules.

Nothing here is specific to one architecture. A model's processor and chat template decide how an
image enters the model (ViT tokens through a projector, cross-attention, discrete image tokens,
raw patches); this module only measures and calls them:
- lengths are counted by encoding with the processor, not by a per-model formula;
- the vision modules are found by name (and can be named by the user), and may not exist at all;
- token ids that a model can write but a text answer must never contain (image placeholders,
  image-generation tokens) are suppressed during generation.
"""

from __future__ import annotations

import re
from typing import Any, Sequence

from . import images

# names of submodules that encode or tokenize images, and of those that map them into the LM
VISION_NAMES = ("visual", "vision_tower", "vision_model", "vision_encoder", "image_encoder", "vqmodel",
                "vision_embed_tokens", "vision_backbone")
PROJECTOR_NAMES = ("multi_modal_projector", "mm_projector", "merger", "connector", "vision_projection",
                   "visual_projection", "aligner")
# config fields naming tokens that the processor replaces with image features
PLACEHOLDER_FIELDS = ("image_token_id", "image_token_index", "video_token_id", "video_token_index")


def load_config(model: str, trust_remote_code: bool = False) -> Any:
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(model, trust_remote_code=trust_remote_code)


def is_vision_config(config: Any) -> bool:
    """A model that takes images: it has a vision (or image tokenizer) sub-config, or transformers
    lists its type among image-text-to-text models."""
    if any(getattr(config, k, None) is not None for k in ("vision_config", "vq_config", "vision_tower")):
        return True
    try:
        from transformers.models.auto.modeling_auto import MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES
    except ImportError:  # pragma: no cover - very old transformers
        return False
    return getattr(config, "model_type", None) in MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES


def is_vision(model: str, trust_remote_code: bool = False) -> bool:
    return is_vision_config(load_config(model, trust_remote_code))


def load_processor(model: str, chat_template: str | None = None, trust_remote_code: bool = False) -> Any:
    """The model's processor, with a padding token and (optionally) another chat template."""
    from transformers import AutoProcessor

    proc = AutoProcessor.from_pretrained(model, trust_remote_code=trust_remote_code)
    tok = tokenizer_of(proc)
    if chat_template:
        if chat_template == "plain":
            raise SystemExit("chat_template 'plain' is for text models; a vision model needs a template that "
                             "places its images (the model's own, a Jinja file or a Jinja string)")
        from distillops.chat import resolve_template

        proc.chat_template = resolve_template(chat_template)
    if not getattr(proc, "chat_template", None):
        if getattr(tok, "chat_template", None):
            proc.chat_template = tok.chat_template
        else:
            raise SystemExit(f"{model} has no chat template; pass chat_template (a Jinja file or string) that "
                             "places the images")
    if tok.eos_token is None:
        raise SystemExit(f"{model}'s tokenizer has no EOS token: generation would never stop")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return proc


def tokenizer_of(processor: Any) -> Any:
    return getattr(processor, "tokenizer", processor)


def load_model(model: str, dtype: Any = None, device_map: Any = None, trust_remote_code: bool = False) -> Any:
    """The model with its generation head, whatever its architecture: the class its config names,
    else the image-text-to-text, vision-to-sequence or causal-LM auto class."""
    import transformers

    config = load_config(model, trust_remote_code)
    kwargs = {"dtype": dtype, "device_map": device_map, "trust_remote_code": trust_remote_code}
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    names = list(getattr(config, "architectures", None) or [])
    candidates = [getattr(transformers, n) for n in names if hasattr(transformers, n)]
    candidates += [getattr(transformers, n) for n in ("AutoModelForImageTextToText", "AutoModelForVision2Seq",
                                                      "AutoModelForCausalLM") if hasattr(transformers, n)]
    errors = []
    for cls in candidates:
        try:
            return cls.from_pretrained(model, **kwargs)
        except (ValueError, KeyError, TypeError) as e:  # this class does not take this config
            errors.append(f"{cls.__name__}: {e}")
    raise SystemExit(f"cannot load {model} as a generative model:\n  " + "\n  ".join(errors))


class TemplateView:
    """The processor's chat template behind the tokenizer's interface, so the framework's text
    helpers (end-of-turn and stop tokens, untrained-token checks) work on a vision model unchanged:
    `apply_chat_template` takes plain string messages, as those helpers send them, and renders them
    with the processor's template; every other attribute is the tokenizer's."""

    def __init__(self, processor: Any):
        self._proc = processor
        self._tok = tokenizer_of(processor)

    def apply_chat_template(self, messages: list[dict], tokenize: bool = True, add_generation_prompt: bool = False,
                            **kwargs: Any) -> Any:
        text = render(self._proc, messages, add_generation_prompt=add_generation_prompt, **kwargs)
        if not tokenize:
            return text
        return self._tok(text, add_special_tokens=False)["input_ids"]

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._tok(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._tok, name)


def render(processor: Any, messages: list[dict], add_generation_prompt: bool = False, **template_kwargs: Any) -> str:
    """The prompt text, with the template's image placeholders where the images go."""
    return processor.apply_chat_template(images.for_template(messages), tokenize=False,
                                         add_generation_prompt=add_generation_prompt, **template_kwargs)


def encode(processor: Any, texts: Sequence[str], part_lists: Sequence[Sequence[dict]], max_pixels: int | None = None,
           **kwargs: Any) -> Any:
    """Processor inputs for a batch: texts from `render`, each example's image parts (sized by
    their own max_pixels, else `max_pixels`). Images go grouped per example (a list per text):
    every processor accepts that form, and some (e.g. Mllama) accept nothing else. The rendered
    template already holds its special tokens (BOS included), so none are added, as in
    distillops' text path and TRL's GRPO."""
    kwargs.setdefault("add_special_tokens", False)
    nested = [[images.load_part(p, max_pixels) for p in parts] for parts in part_lists]
    if any(nested):
        return processor(text=list(texts), images=nested, return_tensors="pt", **kwargs)
    return processor(text=list(texts), return_tensors="pt", **kwargs)


def lengths(processor: Any, row: dict, max_pixels: int | None, **template_kwargs: Any) -> tuple[int, int, bool]:
    """(prompt tokens, prompt + completion tokens, whether the full sequence starts with the prompt)
    of one training row, images included, as the processor actually encodes them."""
    parts = images.image_parts(row["prompt"])
    prompt = render(processor, row["prompt"], add_generation_prompt=True, **template_kwargs)
    full = render(processor, row["prompt"] + row["completion"], **template_kwargs)
    p_ids = encode(processor, [prompt], [parts], max_pixels)["input_ids"][0].tolist()
    f_ids = encode(processor, [full], [parts], max_pixels)["input_ids"][0].tolist()
    return len(p_ids), len(f_ids), f_ids[:len(p_ids)] == p_ids


def prompt_tokens(processor: Any, messages: list[dict], max_pixels: int | None, **template_kwargs: Any) -> int:
    text = render(processor, messages, add_generation_prompt=True, **template_kwargs)
    return int(encode(processor, [text], [images.image_parts(messages)], max_pixels)["input_ids"].shape[1])


def placeholder_ids(model_or_config: Any, processor: Any = None) -> set[int]:
    """Token ids the processor replaces with image (or video) features: their embeddings are never
    read, so they are neither checked for training nor ever written in an answer."""
    config = getattr(model_or_config, "config", model_or_config)
    ids = set()
    for cfg in (config, getattr(config, "text_config", None)):
        for f in PLACEHOLDER_FIELDS:
            v = getattr(cfg, f, None) if cfg is not None else None
            if isinstance(v, int) and v >= 0:
                ids.add(v)
    tok = tokenizer_of(processor) if processor is not None else None
    for attr in ("image_token", "video_token"):
        t = getattr(processor, attr, None) if processor is not None else None
        if isinstance(t, str) and tok is not None:
            i = tok.convert_tokens_to_ids(t)
            if isinstance(i, int) and i != getattr(tok, "unk_token_id", None):
                ids.add(i)
    return ids


def image_generation_ids(model: Any) -> list[int]:
    """Vocabulary ids of image tokens in models that can also write images (discrete image tokens,
    e.g. Chameleon, Emu3); empty for models that only read them."""
    for holder in (model, getattr(model, "model", None)):
        vm = getattr(holder, "vocabulary_mapping", None)
        tokens = getattr(vm, "image_tokens", None) if vm is not None else None
        if tokens is not None:
            return sorted(int(t) for t in tokens)
    return []


def suppressed_ids(model: Any, processor: Any) -> list[int]:
    """Tokens a text answer must never contain."""
    return sorted(placeholder_ids(model, processor) | set(image_generation_ids(model)))


def _matching(model: Any, names: Sequence[str]) -> list[str]:
    """Top-level paths of submodules whose last name is in `names` (a match inside a match is skipped)."""
    found: list[str] = []
    for path, _ in model.named_modules():
        if path and path.split(".")[-1] in names and not any(path.startswith(f + ".") for f in found):
            found.append(path)
    return found


def vision_modules(model: Any, patterns: Sequence[str] | None = None) -> list[str]:
    """The submodules that read images (`patterns`: regexes over module paths, to override)."""
    if patterns:
        regs = [re.compile(p) for p in patterns]
        paths = [p for p, _ in model.named_modules() if p and any(r.search(p) for r in regs)]
        return [p for p in paths if not any(p.startswith(q + ".") for q in paths if q != p)]
    return _matching(model, VISION_NAMES)


def projector_modules(model: Any) -> list[str]:
    return _matching(model, PROJECTOR_NAMES)


def freeze_vision(model: Any, train_vision: bool = False, train_projector: bool = True,
                  patterns: Sequence[str] | None = None) -> dict[str, Any]:
    """Freeze the vision modules (unless `train_vision`) and, unless `train_projector`, the projector.
    A projector nested inside a vision module follows `train_projector`. Returns what was done."""
    vis, proj = vision_modules(model, patterns), projector_modules(model)
    frozen = 0
    for path, param in model.named_parameters():
        in_vis = any(path.startswith(v + ".") for v in vis)
        in_proj = any(path.startswith(p + ".") for p in proj)
        if in_proj:
            train = train_projector
        elif in_vis:
            train = train_vision
        else:
            continue
        if not train:
            param.requires_grad_(False)
            frozen += param.numel()
    if not vis:
        note = ("no separate vision module found (e.g. an encoder-free model); train_vision does not apply"
                if not patterns else f"no module matches vision_modules {list(patterns)}")
        print(f"note: {note}", flush=True)
    return {"vision_modules": vis, "projector_modules": proj, "frozen_params": frozen}


def batches_mixed_ok(processor: Any, with_image: list[dict], text_only: list[dict]) -> bool:
    """Whether the processor accepts one batch holding a prompt with images and a text-only one
    (Mllama's does not)."""
    texts = [render(processor, m, add_generation_prompt=True) for m in (with_image, text_only)]
    try:
        encode(processor, texts, [images.image_parts(with_image), []], padding=True)
    except ValueError:
        return False
    return True


def upcast_trainable(model: Any) -> int:
    """fp32 for trainable parameters that PEFT left in the load dtype (modules_to_save, e.g. the
    projector next to LoRA): fp16 gradients cannot be unscaled, and bf16 weights round small updates
    away. Each module holding a cast parameter casts its floating inputs to fp32 and its outputs
    back to the input dtype, so it still runs where nothing autocasts (TRL's GRPO generation).
    Returns how many parameters were cast."""
    import torch

    def floats(x: Any, dtype: Any) -> Any:
        if torch.is_tensor(x):
            return x.to(dtype) if x.is_floating_point() and x.dtype != dtype else x
        if isinstance(x, (tuple, list)):
            return type(x)(floats(v, dtype) for v in x)
        return x

    def pre(mod: Any, args: tuple, kwargs: dict) -> tuple:
        first = next((t for t in list(args) + list(kwargs.values()) if torch.is_tensor(t) and t.is_floating_point()),
                     None)
        mod._io_dtype = first.dtype if first is not None else None
        return floats(args, torch.float32), {k: floats(v, torch.float32) for k, v in kwargs.items()}

    def post(mod: Any, args: tuple, kwargs: dict, out: Any) -> Any:
        dtype = getattr(mod, "_io_dtype", None)
        return out if dtype in (None, torch.float32) or torch.is_autocast_enabled() else floats(out, dtype)

    n = 0
    for mod in model.modules():
        cast = False
        for p in mod.parameters(recurse=False):
            if p.requires_grad and p.dtype in (torch.float16, torch.bfloat16):
                p.data = p.data.float()
                n += p.numel()
                cast = True
        if cast:
            mod.register_forward_pre_hook(pre, with_kwargs=True)
            mod.register_forward_hook(post, with_kwargs=True)
    return n


def align_inputs(processor: Any, model: Any) -> None:
    """Make the processor produce only inputs the model takes: some tokenizers add token_type_ids,
    which models without that input reject in generate() (TRL's GRPO passes them through too)."""
    import inspect

    tok = tokenizer_of(processor)
    names = list(getattr(tok, "model_input_names", []) or [])
    if "token_type_ids" in names and "token_type_ids" not in inspect.signature(model.forward).parameters:
        tok.model_input_names = [n for n in names if n != "token_type_ids"]


def float_inputs_to(batch: Any, dtype: Any) -> Any:
    """Cast floating-point inputs (pixel values) to the model's dtype; ids and masks stay as they are."""
    for k, v in list(batch.items()):
        if hasattr(v, "is_floating_point") and v.is_floating_point():
            batch[k] = v.to(dtype)
    return batch
