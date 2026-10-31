"""GRPO across TRL versions: TRL 0.28 removed `max_prompt_length`, so the prompt limit is kept by
the framework there, and a real (tiny) GRPO step runs on whatever TRL is installed."""

import dataclasses

import pytest

from distillops.grpo import prompt_length_args


@dataclasses.dataclass
class _OldConfig:  # TRL < 0.28
    max_prompt_length: int | None = 512
    max_completion_length: int = 256


@dataclasses.dataclass
class _NewConfig:  # TRL >= 0.28
    max_completion_length: int = 256
    vllm_max_model_length: int | None = None


class _Tok:
    def __call__(self, text):
        return {"input_ids": text.split()}

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        text = " ".join(m["content"] for m in messages) + (" <assistant>" if add_generation_prompt else "")
        return text.split()


ROWS = [{"prompt": "a b c", "ex_id": "short"}, {"prompt": "a b c d e f", "ex_id": "long"},
        {"prompt": [{"role": "user", "content": "a b"}], "ex_id": "chat"}]  # 2 words + <assistant>


def test_older_trl_gets_the_limit_and_every_row():
    args, rows = prompt_length_args(_OldConfig, ROWS, _Tok(), 4, 16)
    assert args == {"max_prompt_length": 4} and rows == ROWS  # TRL truncates: unchanged behaviour


def test_newer_trl_drops_longer_prompts_and_sizes_vllm(capsys):
    args, rows = prompt_length_args(_NewConfig, ROWS, _Tok(), 4, 16)
    assert args == {"vllm_max_model_length": 20}
    assert [r["ex_id"] for r in rows] == ["short", "chat"]
    assert "1 of 3 prompts are longer than max_prompt_length (4 tokens)" in capsys.readouterr().out
    assert prompt_length_args(_NewConfig, ROWS, _Tok(), None, 16) == ({}, ROWS)  # no limit set
    with pytest.raises(SystemExit, match="every prompt is longer"):
        prompt_length_args(_NewConfig, ROWS, _Tok(), 1, 16)


def test_grpo_trains_a_step_with_the_installed_trl(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("trl")
    import transformers

    from distillops.grpo import GRPORunConfig, train
    from tests.test_any_model import _byte_level_tokenizer

    tok = _byte_level_tokenizer("{% for m in messages %}User:\n{{ m['content'] }}\n\n{% endfor %}"
                                "{% if add_generation_prompt %}Assistant:\n{% endif %}")
    model_dir = tmp_path / "model"
    transformers.GPT2LMHeadModel(transformers.GPT2Config(
        n_layer=1, n_head=2, n_embd=16, n_positions=128, vocab_size=len(tok),
        bos_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id)).save_pretrained(model_dir)
    tok.save_pretrained(model_dir)
    rows = [{"prompt": [{"role": "user", "content": "What is 2+3?"}], "ex_id": str(i)} for i in range(4)]
    rows.append({"prompt": [{"role": "user", "content": "What is 2+3? " * 20}], "ex_id": "too_long"})

    def reward(completions, ex_id, **kwargs):
        return [float(len(c[0]["content"] if isinstance(c, list) else c)) for c in completions]

    cfg = GRPORunConfig(model=str(model_dir), output_dir=str(tmp_path / "out"), num_generations=2,
                        max_prompt_length=32, max_completion_length=4, batch_size=2, grad_accum=1,
                        max_steps=1, use_vllm=False, logging_steps=1, save_steps=1000, eval_every=0)
    out = train(cfg, [reward], rows=rows)
    assert out["steps"] == 1
    import dataclasses as dc

    from trl import GRPOConfig

    if "max_prompt_length" in {f.name for f in dc.fields(GRPOConfig)}:
        assert out["rows"] == 5  # TRL truncates the long prompt itself
    else:
        assert out["rows"] == 4  # left out by the framework
    assert (tmp_path / "out" / "config.json").exists()
