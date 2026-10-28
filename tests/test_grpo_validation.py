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
    score = make_evaluator(GSM8KTask(), exs, max_new_tokens=8, batch_size=2)(model, tok)
    assert score == pytest.approx(3 / 5)  # questions 0, 2, 4 are right
    assert model.training and tok.padding_side == "right"  # state restored after evaluation
