"""Any causal LM as the student: config field names of other architectures, and models without a
chat template (base models such as gpt2)."""

from types import SimpleNamespace

import pytest

from distillops.chat import PLAIN_TEMPLATE, chat_ids, prepare_tokenizer, resolve_template
from distillops.memory import kv_bytes_per_token, max_sequences, model_dims
from distillops.preflight import model_bytes

GPT2 = SimpleNamespace(n_layer=12, n_head=12, n_embd=768, vocab_size=50257)
LLAMA_GQA = SimpleNamespace(num_hidden_layers=32, num_attention_heads=32, num_key_value_heads=8,
                            hidden_size=4096, vocab_size=128256)


def test_dims_under_each_architectures_field_names():
    assert model_dims(GPT2) == {"layers": 12, "heads": 12, "kv_heads": 12, "hidden": 768, "head_dim": 64,
                                "vocab": 50257}
    assert model_dims(LLAMA_GQA)["kv_heads"] == 8 and model_dims(LLAMA_GQA)["head_dim"] == 128
    chatglm = SimpleNamespace(num_layers=28, num_attention_heads=32, hidden_size=4096, kv_channels=128,
                              multi_query_attention=True, multi_query_group_num=2, padded_vocab_size=65024)
    assert model_dims(chatglm) == {"layers": 28, "heads": 32, "kv_heads": 2, "hidden": 4096, "head_dim": 128,
                                   "vocab": 65024}


def test_multi_query_and_explicit_kv_heads():
    falcon7b = SimpleNamespace(num_hidden_layers=32, num_attention_heads=71, num_kv_heads=71, hidden_size=4544,
                               multi_query=True, new_decoder_architecture=False, vocab_size=65024)
    falcon40b = SimpleNamespace(num_hidden_layers=60, num_attention_heads=128, num_kv_heads=8, hidden_size=8192,
                                multi_query=True, new_decoder_architecture=True, vocab_size=65024)
    assert model_dims(falcon7b)["kv_heads"] == 1
    assert model_dims(falcon40b)["kv_heads"] == 8


def test_multimodal_config_uses_its_language_model():
    vlm = SimpleNamespace(text_config=LLAMA_GQA, vision_config=SimpleNamespace(hidden_size=1152))
    assert model_dims(vlm) == model_dims(LLAMA_GQA)


def test_kv_size_for_gpt2_names():
    assert kv_bytes_per_token(GPT2) == 2 * 12 * 12 * 64 * 2


def test_unknown_architecture_falls_back_to_a_small_batch():
    with pytest.raises(ValueError):
        model_dims(SimpleNamespace(d_state=16, vocab_size=50280))  # e.g. a state-space model
    model = SimpleNamespace(config=SimpleNamespace(d_state=16))
    assert max_sequences(model, 512, free_bytes=80 * 2**30) == 8
    assert max_sequences(SimpleNamespace(config=GPT2), 512, free_bytes=8 * 2**30, cap=10_000) > 8


def test_model_bytes_from_gpt2_style_config(monkeypatch, tmp_path):
    transformers = pytest.importorskip("transformers")
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", staticmethod(lambda m: GPT2))
    n = model_bytes(str(tmp_path / "no-weights-here"))
    assert 0.8 * 124e6 * 2 < n < 1.6 * 124e6 * 2  # gpt2 has 124M parameters, 2 bytes each


def test_resolve_template(tmp_path):
    assert resolve_template("plain") == PLAIN_TEMPLATE
    f = tmp_path / "t.jinja"
    f.write_text("{{ messages[0]['content'] }}", encoding="utf-8")
    assert resolve_template(str(f)) == "{{ messages[0]['content'] }}"
    assert resolve_template("{% for m in messages %}{{ m['content'] }}{% endfor %}").startswith("{%")
    with pytest.raises(ValueError):
        resolve_template("missing-file.jinja")


def _byte_level_tokenizer(chat_template=None):
    """A gpt2-style (byte-level BPE) tokenizer built in memory, with no chat template."""
    pytest.importorskip("transformers")
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    tk = Tokenizer(models.BPE())
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=400, special_tokens=["<eos>"],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tk.train_from_iterator(["User:\nWhat is 2+3?\n\nAssistant:\n2+3 = 5. The answer is 5.\n"] * 50, trainer)
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="<eos>")
    tok.chat_template = chat_template
    return tok


CONVERSATIONS = [
    ([{"role": "user", "content": "What is 2+3?"}], "2+3 = 5. The answer is 5."),
    ([{"role": "system", "content": "Be terse."}, {"role": "user", "content": " spaced\tinput\n"}], "5"),
    ([{"role": "user", "content": "What is 2+3?"}], "\n\nleading newlines and a trailing space "),
]


@pytest.mark.parametrize("prompt,answer", CONVERSATIONS)
def test_plain_template_for_a_model_without_one(prompt, answer):
    tok = prepare_tokenizer(_byte_level_tokenizer())
    assert tok.chat_template == PLAIN_TEMPLATE and tok.pad_token == "<eos>"
    completion = [{"role": "assistant", "content": answer}]
    prompt_ids = chat_ids(tok, prompt, add_generation_prompt=True)
    ids = chat_ids(tok, prompt + completion)
    assert ids[: len(prompt_ids)] == prompt_ids  # completion-only loss masks exactly the prompt
    assert ids[-1] == tok.eos_token_id  # the model learns to stop
    assert tok.decode(ids[len(prompt_ids):]) == answer.strip() + "<eos>"
    rendered = tok.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    assert rendered.endswith("Assistant:\n")  # Qwen-style template options are ignored, not an error


def test_own_template_is_kept_and_an_explicit_one_wins():
    own = "{% for m in messages %}<{{ m['role'] }}>{{ m['content'] }}{% endfor %}"
    assert prepare_tokenizer(_byte_level_tokenizer(own)).chat_template == own
    assert prepare_tokenizer(_byte_level_tokenizer(own), "plain").chat_template == PLAIN_TEMPLATE


def test_tokenizer_without_eos_is_refused():
    tok = SimpleNamespace(chat_template="x", eos_token=None, pad_token=None)
    with pytest.raises(ValueError, match="EOS"):
        prepare_tokenizer(tok)


def test_chat_ids_across_return_types():
    class Tok:
        def __init__(self, out):
            self.out = out

        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            return self.out

    assert chat_ids(Tok([1, 2]), []) == [1, 2]
    assert chat_ids(Tok({"input_ids": [1, 2], "attention_mask": [1, 1]}), []) == [1, 2]  # transformers 5
    assert chat_ids(Tok([[1, 2]]), []) == [1, 2]  # multimodal processors


def test_cli_accepts_model_and_template_options(monkeypatch):
    import distillops.cli as cli

    seen = {}
    monkeypatch.setattr(cli, "cmd_grpo", lambda a: seen.update(grpo=(a.model, a.chat_template)))
    monkeypatch.setattr(cli, "cmd_sft", lambda a: seen.update(sft=a.chat_template))
    monkeypatch.setattr(cli, "cmd_eval", lambda a: seen.update(eval=a.chat_template))
    for cmd in (["grpo", "c.yaml", "--task", "t:T", "--model", "gpt2", "--chat-template", "plain"],
                ["sft", "c.yaml", "--chat-template", "plain"],
                ["eval", "--task", "t:T", "--model", "gpt2", "--chat-template", "plain"]):
        cli.main(cmd)
    assert seen == {"grpo": ("gpt2", "plain"), "sft": "plain", "eval": "plain"}
