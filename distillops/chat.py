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
