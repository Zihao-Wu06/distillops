"""Validation-based checkpoint selection for GRPO: split, evaluator and selector logic."""

from types import SimpleNamespace

import pytest

from distillops.grpo import split_validation
from tasks.math.gsm8k import GSM8KTask, MathExample

EXS = [MathExample(f"gsm8k:train:{i}", f"q{i}", "", str(i)) for i in range(50)]


class NoValid(GSM8KTask):
    def load_examples(self, split):
        if split == "valid":
            raise ValueError("Unknown split 'valid'")  # what datasets raises for GSM8K
        return EXS


class WithValid(GSM8KTask):
    VALID = [MathExample(f"gsm8k:valid:{i}", f"v{i}", "", str(i)) for i in range(30)]

    def load_examples(self, split):
        return self.VALID if split == "valid" else EXS


def test_split_uses_the_valid_split_and_leaves_training_untouched():
    train, valid, source = split_validation(WithValid(), EXS, 10, seed=0)
    assert source == "valid split" and train == EXS and len(valid) == 10
    assert all(ex.id.startswith("gsm8k:valid") for ex in valid)


def test_split_holds_out_disjoint_prompts_when_there_is_no_valid_split():
    train, valid, source = split_validation(NoValid(), EXS, 200, seed=0)
    assert source == "held out of training"
    assert len(valid) == 10  # capped at 20% of 50
    assert {e.id for e in train}.isdisjoint({e.id for e in valid}) and len(train) + len(valid) == 50
    again, valid2, _ = split_validation(NoValid(), EXS, 200, seed=0)
    assert [e.id for e in valid2] == [e.id for e in valid]  # deterministic: same prompts held out every run


def test_split_disabled():
    assert split_validation(NoValid(), EXS, 0) == (EXS, [], "none")


def _run_selector(scores, every=2, patience=0, total_steps=7):
    pytest.importorskip("transformers")
    from distillops.grpo import validation_selector

    it = iter(scores)
    saved = []
    state = SimpleNamespace(global_step=0)
    sel = validation_selector(lambda model: next(it), every, patience, lambda: saved.append(state.global_step))
    control = SimpleNamespace(should_training_stop=False)
    sel.on_train_begin(None, state, control, model=None)
    for step in range(1, total_steps + 1):
        state.global_step = step
        sel.on_step_end(None, state, control, model=None)
        if control.should_training_stop:
            break
    sel.on_train_end(None, state, control, model=None)
    return sel, saved, state.global_step


def test_selector_keeps_the_best_step_and_saves_only_improvements():
    # checks at steps 0, 2, 4, 6 and a final check at 7
    sel, saved, _ = _run_selector([0.50, 0.55, 0.60, 0.58, 0.59])
    assert [s for s, _ in sel.history] == [0, 2, 4, 6, 7]
    assert sel.best_step == 4 and sel.best == 0.60 and saved == [2, 4]


def test_selector_reports_the_start_when_nothing_beats_it():
    sel, saved, _ = _run_selector([0.50, 0.40, 0.45, 0.49, 0.30])
    assert sel.best_step == 0 and saved == []  # the caller then delivers the starting model


def test_selector_stops_early_after_patience_checks():
    sel, saved, last = _run_selector([0.5, 0.6, 0.55, 0.54, 0.9], patience=2, total_steps=20)
    assert last == 6 and sel.best_step == 2 and saved == [2]
    assert [s for s, _ in sel.history] == [0, 2, 4, 6]  # the final check at step 6 is not repeated


def test_selector_off():
    sel, saved, _ = _run_selector([], every=0)
    assert sel.history == [] and saved == []


def test_evaluator_scores_greedy_outputs_and_restores_state():
    torch = pytest.importorskip("torch")
    from distillops.grpo import make_evaluator

    exs = EXS[:5]
    texts = {}

    class Tok:
        padding_side, pad_token, eos_token, pad_token_id = "right", "<pad>", "<eos>", 0

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            return messages[0]["content"]

        def __call__(self, batch, return_tensors=None, padding=None, add_special_tokens=False):
            if isinstance(batch, str):
                return {"input_ids": [1]}
            ids = torch.tensor([[texts.setdefault(t, len(texts) + 1)] for t in batch])
            enc = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
            return SimpleNamespace(**enc, to=lambda device: enc)

        def decode(self, row, skip_special_tokens=True):
            i = int(row[0]) - 1000  # the fake model echoes the prompt id + 1000
            q = next(t for t, k in texts.items() if k == i)
            n = int(q.split()[0][1:])
            return f"ANSWER: {n if n % 2 == 0 else -1}"  # right on even questions only

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(1))

        def generate(self, input_ids, attention_mask=None, **kw):
            assert not self.training  # generation must run in eval mode
            return torch.cat([input_ids, input_ids + 1000], 1)

    tok, model = Tok(), Model()
    model.train()
    score, se = make_evaluator(GSM8KTask(), exs, max_new_tokens=8, batch_size=2)(model, tok)
    assert score == pytest.approx(3 / 5)  # questions 0, 2, 4 are right
    assert se == pytest.approx((0.3 / 5) ** 0.5)  # sample variance 0.3 over 5 questions
    assert model.training and tok.padding_side == "right"  # state restored after evaluation


# -- validation never uses prompts the start model was trained on (validation leakage) ------------

def test_validation_leaves_out_prompts_the_start_model_learned():
    seen = {e.id for e in EXS[:40]}  # SFT used 40 of the 50 prompts
    train, valid, source = split_validation(NoValid(), EXS, 200, seed=0, exclude_ids=seen)
    assert valid and not {e.id for e in valid} & seen and "left out" in source
    assert {e.id for e in train}.isdisjoint({e.id for e in valid}) and len(train) + len(valid) == 50
    train, valid, source = split_validation(NoValid(), EXS, 200, seed=0, exclude_ids={e.id for e in EXS})
    assert valid == [] and train == EXS and source.startswith("none: every prompt")  # all seen: no validation
    assert split_validation(NoValid(), EXS, 200, seed=0, exclude_ids=None)[1]  # no record: as before


def test_valid_split_also_drops_prompts_seen_in_training():
    seen = {f"gsm8k:valid:{i}" for i in range(25)}
    train, valid, source = split_validation(WithValid(), EXS, 10, seed=0, exclude_ids=seen)
    assert train == EXS and len(valid) == 5 and not {e.id for e in valid} & seen and "25 seen" in source
    _, valid, source = split_validation(WithValid(), EXS, 10, exclude_ids={e.id for e in WithValid.VALID})
    assert valid == [] and source.startswith("none")


def test_validation_log_shows_the_noise(capsys):
    from distillops.grpo import mean_and_se, validation_selector

    assert mean_and_se([]) == (0.0, 0.0) and mean_and_se([1.0]) == (1.0, 0.0)
    sel = validation_selector(lambda model: (0.91, 0.0202), 25, 0, lambda: None)
    sel._check(0, None)
    assert "0.9100 ± 0.0202" in capsys.readouterr().out and sel.history == [(0, 0.91)]


def test_trained_ids_round_trip(tmp_path):
    from distillops.task import read_trained_ids, write_trained_ids

    assert read_trained_ids(tmp_path) is None
    write_trained_ids(tmp_path, ["b", "a", "a"])  # not a model folder: nothing written
    assert read_trained_ids(tmp_path) is None
    (tmp_path / "config.json").write_text("{}")
    write_trained_ids(tmp_path, ["b", "a", "a"])
    assert read_trained_ids(tmp_path) == {"a", "b"}


def test_sft_length_filter_reports_the_ids_it_keeps():
    from tests.test_any_model import _byte_level_tokenizer

    from distillops.chat import prepare_tokenizer
    from distillops.sft import length_filter

    tok = prepare_tokenizer(_byte_level_tokenizer())
    rows = [{"prompt": [{"role": "user", "content": "q"}], "completion": [{"role": "assistant", "content": c}],
             "meta": {"id": f"r{i}"}} for i, c in enumerate(["short", "x " * 500, "fine"])]
    ids: list = []
    kept, _ = length_filter(rows, tok, max_length=64, ids_out=ids)
    assert len(kept) == 2 and ids == ["r0", "r2"] and "meta" not in kept[0]


def test_labels_rows_carry_their_id(tmp_path, monkeypatch):
    import json

    from distillops import cli

    monkeypatch.setattr("tests.test_framework.EvenTask.load_examples", lambda self, split: [1, 2], raising=False)
    monkeypatch.setattr("tests.test_framework.EvenTask.gold_sft_example",
                        lambda self, ex: {"prompt": [], "completion": [{"role": "assistant", "content": str(2 * ex)}]},
                        raising=False)
    out = tmp_path / "labels.jsonl"
    cli.main(["labels", "--task", "tests.test_framework:EvenTask", "--out", str(out)])
    assert [json.loads(line)["meta"]["id"] for line in out.read_text().splitlines()] == ["1", "2"]


def test_gsm8k_valid_size_sets_training_problems_apart(monkeypatch):
    import datasets

    rows = [{"question": f"q{i}", "answer": f"s #### {i}"} for i in range(10)]
    monkeypatch.setattr(datasets, "load_dataset", lambda *a, split, **k: rows if split == "train" else rows[:3])
    plain, held = GSM8KTask(), GSM8KTask(valid_size=4)
    assert len(plain.load_examples("train")) == 10
    train, valid = held.load_examples("train"), held.load_examples("valid")
    assert len(train) == 6 and len(valid) == 4 and {e.id for e in train}.isdisjoint({e.id for e in valid})
    assert valid[0].id == "gsm8k:train:6"  # ids keep their train index (teacher logs and caches still match)
    assert [e.id for e in held.load_examples("test")] == ["gsm8k:test:0", "gsm8k:test:1", "gsm8k:test:2"]
    with pytest.raises(ValueError):
        GSM8KTask(valid_size=10).load_examples("train")


def test_grpo_warns_when_a_local_start_model_has_no_record(tmp_path, monkeypatch, capsys):
    from distillops import cli

    start = tmp_path / "sft_model"  # an SFT model made before trained_ids.txt existed
    start.mkdir()
    cfg = tmp_path / "g.yaml"
    cfg.write_text(f"model: {start}\noutput_dir: {tmp_path / 'out'}\n")
    monkeypatch.setattr("distillops.grpo.train", lambda *a, **k: {})
    monkeypatch.setattr("distillops.grpo.make_evaluator", lambda *a, **k: None)
    monkeypatch.setattr(NoValid, "load_examples", lambda self, split: EXS if split == "train" else [], raising=False)
    monkeypatch.setattr("tests.test_grpo_validation.NoValid.student_messages", lambda self, ex: [{"role": "user", "content": "q"}],
                        raising=False)
    cli.main(["grpo", str(cfg), "--task", "tests.test_grpo_validation:NoValid", "--eval-examples", "5"])
    assert "has no record of the examples it was trained on" in capsys.readouterr().out
