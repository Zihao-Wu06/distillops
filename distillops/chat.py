"""The student's tokenizer and chat template, for any model.

Training, evaluation and RL all turn chat messages into text with the tokenizer's chat template,
so the student is trained and served on the same format. Instruct models ship a template; base
models (e.g. gpt2, many pretrained checkpoints) do not. Choose one with `chat_template`:

- unset: the model's own template, or PLAIN_TEMPLATE if it has none (a notice is printed);
- "plain": PLAIN_TEMPLATE even if the model has its own;
- a path to a Jinja file, or a Jinja string: that template.

The template is saved with every trained model, so later stages pick it up without the option.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# Role-labelled turns; each assistant turn ends with the EOS token so the model learns to stop.
# Assistant text is trimmed: a reply starting with a newline would merge with the newline that
# ends the generation prompt into one token, and completion-only loss needs the prompt's tokens to
# be a prefix of prompt+completion's tokens.
PLAIN_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'assistant' %}"
    "{{ 'Assistant:\n' + (message['content'] | trim) + eos_token }}"
    "{% if not loop.last %}{{ '\n\n' }}{% endif %}"
    "{% else %}"
    "{{ message['role'] | capitalize }}{{ ':\n' + message['content'] + '\n\n' }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ 'Assistant:\n' }}{% endif %}"
)


def resolve_template(chat_template: str) -> str:
    if chat_template == "plain":
        return PLAIN_TEMPLATE
    p = Path(chat_template)
    if p.is_file():
        return p.read_text(encoding="utf-8")
    if "{{" in chat_template or "{%" in chat_template:
        return chat_template
    raise ValueError(f"chat_template {chat_template!r} is not 'plain', a file, or a Jinja template")


def prepare_tokenizer(tok: Any, chat_template: str | None = None, model: str = "") -> Any:
    """Give `tok` a chat template (see module docstring) and a padding token; returns `tok`."""
    if chat_template:
        tok.chat_template = resolve_template(chat_template)
    elif not getattr(tok, "chat_template", None):
        tok.chat_template = PLAIN_TEMPLATE
        print(f"note: {model or 'the model'} has no chat template; using the plain 'User:/Assistant:' "
              "template (set chat_template to choose another)", flush=True)
    if tok.eos_token is None:
        raise ValueError(f"{model or 'the tokenizer'} has no EOS token: generation would never stop")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_tokenizer(model: str, chat_template: str | None = None) -> Any:
    from transformers import AutoTokenizer

    return prepare_tokenizer(AutoTokenizer.from_pretrained(model), chat_template, model)


def chat_ids(tok: Any, messages: list[dict], add_generation_prompt: bool = False) -> list[int]:
    """Token ids of rendered messages, as TRL computes them (list of ints across transformers versions:
    some return a dict, and multimodal processors a batch of one)."""
    out = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=add_generation_prompt)
    if hasattr(out, "keys"):
        out = out["input_ids"]
    return list(out[0]) if out and isinstance(out[0], list) else list(out)


def device() -> str:
    """Where to run inference: the GPU if there is one, else the CPU."""
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def end_of_turn_id(tok: Any) -> int | None:
    """The token the chat template puts right after an assistant reply (e.g. <|im_end|>), if it is
    a single token other than EOS. A model trained on such turns ends its reply with that token,
    so generation must stop there too; base models often keep a different EOS (<|endoftext|>)."""
    marker = "\x00distillops-end-of-turn\x00"
    try:
        text = tok.apply_chat_template([{"role": "user", "content": "q"}, {"role": "assistant", "content": marker}],
                                       tokenize=False)
    except Exception:  # a template that needs other roles or fields: leave stopping to EOS
        return None
    if marker not in text:
        return None
    tail = text.split(marker, 1)[1].strip()
    if not tail:
        return None
    ids = tok(tail, add_special_tokens=False)["input_ids"]
    if len(ids) < 1:
        return None
    first = ids[0]  # e.g. "<|im_end|>" (then a newline): the first token after the reply
    if tok.convert_ids_to_tokens(first) not in tail or first == tok.eos_token_id:
        return None
    special = set(getattr(tok, "all_special_ids", [])) | set(getattr(tok, "added_tokens_decoder", {}) or {})
    return first if first in special else None


def stop_token_ids(tok: Any, model_eos: Any = None) -> list[int]:
    """EOS, the template's end-of-turn token and any EOS ids the model's generation config lists."""
    ids: list[int] = []
    extra = model_eos if isinstance(model_eos, (list, tuple)) else [model_eos]
    for i in [getattr(tok, "eos_token_id", None), end_of_turn_id(tok), *extra]:
        if isinstance(i, int) and i not in ids:
            ids.append(i)
    return ids


def save_stop_tokens(model_dir: str | Path, tok: Any) -> list[int]:
    """Record the stop tokens in a saved model's generation_config.json, so transformers and vLLM
    (which reads that file) stop where the model was trained to end its reply."""
    from transformers import GenerationConfig

    model_dir = Path(model_dir)
    if not (model_dir / "config.json").exists():  # e.g. a LoRA adapter folder: nothing to generate with
        return []
    if (model_dir / "generation_config.json").exists():
        gen = GenerationConfig.from_pretrained(model_dir)
    else:  # derive bos / eos / pad from the model config, as transformers would
        from transformers import AutoConfig

        try:
            gen = GenerationConfig.from_model_config(AutoConfig.from_pretrained(model_dir))
        except (OSError, ValueError):  # a config transformers cannot read: start from defaults
            gen = GenerationConfig()
    ids = stop_token_ids(tok, gen.eos_token_id)
    gen.eos_token_id = ids if len(ids) > 1 else ids[0]
    if gen.pad_token_id is None and tok.pad_token_id is not None:
        gen.pad_token_id = tok.pad_token_id
    gen.save_pretrained(model_dir)
    return ids


def untrained_token_ids(model: Any, tok: Any, token_ids: Any, written_ids: Any = None) -> list[int]:
    """Which special tokens the rows use have an untrained embedding: all zeros, or identical to
    another special token's row (models initialise unused special tokens to one shared vector).
    Input rows are checked for every token in `token_ids`; a separate (untied) output layer only for
    `written_ids`, the tokens the model must write (default: all), since a token that only ever
    appears in prompts never needs its output row. LoRA trains neither."""
    import torch

    special = sorted(set(getattr(tok, "all_special_ids", [])) | set(getattr(tok, "added_tokens_decoder", {}) or {}))
    used = sorted({int(i) for i in token_ids} & set(special))
    written = used if written_ids is None else sorted({int(i) for i in written_ids} & set(special))
    if not used:
        return []
    bad: set[int] = set()
    inp = model.get_input_embeddings().weight
    checks = [(inp, used)]
    out = model.get_output_embeddings()
    if out is not None and out.weight.data_ptr() != inp.data_ptr():
        checks.append((out.weight, written))
    with torch.no_grad():
        for w, ids in checks:
            rows = w[[i for i in special if i < w.shape[0]]].float()
            index = {i: k for k, i in enumerate(i for i in special if i < w.shape[0])}
            for i in ids:
                if i not in index:
                    continue
                r = rows[index[i]]
                if not bool(r.abs().sum() > 0) or bool((rows == r).all(dim=1).sum() > 1):
                    bad.add(i)
    return sorted(bad)
