"""Base models whose chat template ends a reply with a token other than EOS, or uses special
tokens the model never trained: generation must stop at the end of a turn, and LoRA is refused
when it could not learn those tokens."""

import json
from types import SimpleNamespace

import pytest

from distillops.chat import PLAIN_TEMPLATE, end_of_turn_id, save_stop_tokens, stop_token_ids, untrained_token_ids
from tests.test_any_model import _byte_level_tokenizer

QWEN_STYLE = ("{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
              "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")
CHAT_TOKENS = ["<|im_start|>", "<|im_end|>", "<think>", "</think>"]


def _qwen_style_tokenizer():
    tok = _byte_level_tokenizer(QWEN_STYLE)
    tok.add_special_tokens({"additional_special_tokens": CHAT_TOKENS})
    return tok


def test_end_of_turn_token_found_when_it_is_not_eos():
    tok = _qwen_style_tokenizer()
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    assert end_of_turn_id(tok) == im_end
    assert stop_token_ids(tok) == [tok.eos_token_id, im_end]
    assert stop_token_ids(tok, [tok.eos_token_id, 7]) == [tok.eos_token_id, im_end, 7]  # the model's own EOS ids kept
    assert stop_token_ids(tok, im_end) == [tok.eos_token_id, im_end]  # an int is fine, no duplicates


def test_no_extra_stop_token_when_the_template_ends_with_eos_or_plain_text():
    assert end_of_turn_id(_byte_level_tokenizer(PLAIN_TEMPLATE)) is None  # the plain template ends with EOS
    tok = _byte_level_tokenizer("{% for m in messages %}{{ m['content'] }}\n\n### {% endfor %}")
    assert end_of_turn_id(tok) is None  # ordinary text after the reply: nothing to add
    broken = _byte_level_tokenizer("{{ raise_exception('needs a system message') }}")
    assert end_of_turn_id(broken) is None
    assert stop_token_ids(broken) == [broken.eos_token_id]


def test_stop_tokens_written_to_generation_config(tmp_path):
    pytest.importorskip("transformers")
    from transformers import GenerationConfig

    tok = _qwen_style_tokenizer()
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}")
    GenerationConfig(eos_token_id=tok.eos_token_id, max_new_tokens=99).save_pretrained(model_dir)
    ids = save_stop_tokens(model_dir, tok)
    saved = json.loads((model_dir / "generation_config.json").read_text())
    assert saved["eos_token_id"] == ids == [tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")]
    assert saved["max_new_tokens"] == 99  # other settings kept
    adapter = tmp_path / "adapter"  # a LoRA adapter folder has no config.json: left alone
    adapter.mkdir()
    assert save_stop_tokens(adapter, tok) == [] and not (adapter / "generation_config.json").exists()
    fresh = tmp_path / "fresh"  # a model folder without generation_config.json gets one
    fresh.mkdir()
    (fresh / "config.json").write_text("{}")
    save_stop_tokens(fresh, tok)
    assert json.loads((fresh / "generation_config.json").read_text())["eos_token_id"][1] == ids[1]


def _model_with(rows):
    torch = pytest.importorskip("torch")
    emb = torch.nn.Embedding(len(rows), 3)
    with torch.no_grad():
        emb.weight.copy_(torch.tensor(rows, dtype=torch.float32))
    return SimpleNamespace(get_input_embeddings=lambda: emb, get_output_embeddings=lambda: None)


def test_untrained_special_tokens_are_found():
    tok = SimpleNamespace(all_special_ids=[0, 3, 4, 5], added_tokens_decoder={3: "a", 4: "b", 5: "c"})
    trained = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 0, 2], [0, 3, 1], [2, 2, 5]]
    assert untrained_token_ids(_model_with(trained), tok, [1, 3, 4]) == []
    shared = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [.1, .1, .1], [.1, .1, .1], [.1, .1, .1]]  # one init vector
    assert untrained_token_ids(_model_with(shared), tok, [1, 2, 3, 4]) == [3, 4]  # only the ones the rows use
    zero = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [0, 0, 0], [0, 3, 1], [2, 2, 5]]
    assert untrained_token_ids(_model_with(zero), tok, [3, 4]) == [3]
    assert untrained_token_ids(_model_with(trained), tok, [1, 2]) == []  # ordinary tokens are not checked


def test_lora_is_refused_on_untrained_template_tokens(capsys):
    from distillops.sft import check_template_tokens

    tok = _qwen_style_tokenizer()
    n = len(tok)
    rows = [[float(i), 1.0, 2.0] for i in range(n)]
    for t in CHAT_TOKENS:  # the base model never trained the chat tokens: one shared vector
        rows[tok.convert_tokens_to_ids(t)] = [0.5, 0.5, 0.5]
    model = _model_with(rows)
    row = {"prompt": [{"role": "user", "content": "What is 2+3?"}],
           "completion": [{"role": "assistant", "content": "5"}]}
    with pytest.raises(SystemExit, match="lora: false"):
        check_template_tokens(model, tok, row, lora=True)
    check_template_tokens(model, tok, row, lora=False)  # full fine-tuning trains them: a note only
    assert "full fine-tuning will train them" in capsys.readouterr().out
    trained = _model_with([[float(i), 1.0, float(i % 7)] for i in range(n)])
    check_template_tokens(trained, tok, row, lora=True)  # an instruct model: nothing to say


def test_prompt_only_tokens_need_no_trained_output_row():
    torch = pytest.importorskip("torch")
    tok = SimpleNamespace(all_special_ids=[0, 3, 4], added_tokens_decoder={3: "bos", 4: "reserved"})
    emb = torch.nn.Embedding(5, 3)
    head = torch.nn.Linear(3, 5, bias=False)  # untied output layer
    with torch.no_grad():
        emb.weight.copy_(torch.tensor([[1., 2, 3], [4, 5, 6], [7, 8, 9], [1, 0, 2], [0, 3, 1]]))
        head.weight.copy_(torch.tensor([[1., 2, 3], [4, 5, 6], [7, 8, 9], [.1, .1, .1], [.1, .1, .1]]))
    model = SimpleNamespace(get_input_embeddings=lambda: emb, get_output_embeddings=lambda: head)
    # token 3 is a prompt-only token (e.g. a BOS): its output row never trains, which is fine
    assert untrained_token_ids(model, tok, [3, 1, 2], written_ids=[1, 2]) == []
    assert untrained_token_ids(model, tok, [3, 1, 2], written_ids=[3]) == [3]  # if the model had to write it


def test_stop_tokens_derived_from_the_model_config_when_generation_config_is_missing(tmp_path):
    transformers = pytest.importorskip("transformers")

    tok = _qwen_style_tokenizer()
    d = tmp_path / "m"
    transformers.GPT2Config(n_layer=1, n_head=1, n_embd=8, vocab_size=len(tok), bos_token_id=5,
                            eos_token_id=tok.eos_token_id).save_pretrained(d)
    save_stop_tokens(d, tok)
    saved = json.loads((d / "generation_config.json").read_text())
    assert saved["bos_token_id"] == 5 and tok.convert_tokens_to_ids("<|im_end|>") in saved["eos_token_id"]
