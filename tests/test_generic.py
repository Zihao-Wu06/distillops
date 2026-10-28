"""The generic pieces: task loading, CLI, evaluation and GRPO adapters, and the GSM8K task."""

import json
import random

from distillops.cli import load_task, main, subsample
from distillops.evaluate import evaluate_task
from distillops.grpo import task_rows_and_reward
from tasks.math.gsm8k import GSM8KTask, MathExample, final_answer

EX = MathExample("gsm8k:train:0", "Tom has 3 apples and buys 4 more. How many?",
                 "He has 3+4=<<3+4=7>>7 apples.", "7")
EX_BIG = MathExample("gsm8k:train:1", "Big number?", "", "1,200".replace(",", ""))


def test_load_task_from_path_and_args(tmp_path):
    task = load_task("tasks.math.gsm8k:GSM8KTask", ["max_words=100", "label_reasoning=human"])
    assert isinstance(task, GSM8KTask) and task.max_words == 100 and task.label_reasoning == "human"
    spec = tmp_path / "t.yaml"
    spec.write_text("task: tasks.math.gsm8k:GSM8KTask\nargs:\n  max_words: 120\n")
    assert load_task(str(spec), ["label_reasoning=human"]).max_words == 120


def test_subsample_matches_the_old_eval_scripts():
    xs = list(range(50))
    assert subsample(xs, 10, 0) == random.Random(0).sample(xs, 10)
    assert subsample(xs, 0, 0) == xs


def test_gsm8k_verify():
    task = GSM8KTask()
    good = "3 + 4 = 7\nANSWER: 7"
    assert task.verify(EX, task.parse_teacher(EX, good), good).accepted
    wrong = "3 + 5 = 8\nANSWER: 8"
    assert task.verify(EX, task.parse_teacher(EX, wrong), wrong).reason == "wrong_answer"
    assert task.verify(EX, task.parse_teacher(EX, "it is 7"), "it is 7").reason == "parse_error"
    commas = "lots of steps\nANSWER: $1,200"
    assert task.verify(EX_BIG, task.parse_teacher(EX_BIG, commas), commas).accepted


def test_gsm8k_score_reads_the_answer_after_the_think_block():
    task = GSM8KTask()
    assert task.score(EX, "<think>\n3+4=7, not 8\n</think>\n\nANSWER: 7") == 1.0
    assert task.score(EX, "<think>\nthe answer is 7\n</think>\n\nANSWER: 8") == 0.0
    assert task.score(EX, "3 plus 4 makes 7") == 1.0  # no ANSWER line: last number
    assert final_answer("ANSWER: 7.0") == 7.0


def test_gsm8k_rows_have_the_same_layout_for_teacher_and_labels():
    task = GSM8KTask()
    row = task.sft_example(EX, task.parse_teacher(EX, "3 + 4 = 7\nANSWER: 7"), "")
    assert row["completion"][0]["content"] == "<think>\n3 + 4 = 7\n</think>\n\nANSWER: 7"
    assert row["prompt"] == task.student_messages(EX)
    assert task.gold_sft_example(EX)["completion"][0]["content"] == "<think>\n\n</think>\n\nANSWER: 7"
    human = GSM8KTask(label_reasoning="human").gold_sft_example(EX)["completion"][0]["content"]
    assert "<<" not in human and "3+4=7 apples" in human


def test_evaluate_task_with_a_fake_generator(tmp_path):
    task = GSM8KTask()
    fake = lambda model, prompts, **kw: ["ANSWER: 7", "ANSWER: 5"][: len(prompts)]  # noqa: E731
    m = evaluate_task(task, "fake-model", [EX, EX_BIG], tmp_path, generate_fn=fake, max_new_tokens=64)
    assert m["score"] == 0.5 and m["n"] == 2 and m["max_new_tokens"] == 64
    preds = [json.loads(l) for l in open(tmp_path / "predictions.jsonl", encoding="utf-8")]
    assert [p["score"] for p in preds] == [1.0, 0.0]


def test_grpo_reward_uses_task_score():
    task = GSM8KTask()
    rows, reward = task_rows_and_reward(task, [EX, EX_BIG])
    assert rows[0]["prompt"] == task.student_messages(EX) and rows[1]["ex_id"] == EX_BIG.id
    completions = [[{"role": "assistant", "content": "ANSWER: 7"}], [{"role": "assistant", "content": "ANSWER: 1200"}]]
    assert reward(completions, ex_id=[EX.id, EX_BIG.id]) == [1.0, 1.0]
    assert reward.__name__ == "gsm8k_score"


def test_cli_labels_writes_control_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(GSM8KTask, "load_examples", lambda self, split: [EX, EX_BIG])
    out = tmp_path / "labels.jsonl"
    main(["labels", "--task", "tasks.math.gsm8k:GSM8KTask", "--out", str(out)])
    rows = [json.loads(l) for l in open(out, encoding="utf-8")]
    assert len(rows) == 2 and rows[1]["completion"][0]["content"].endswith("ANSWER: 1200")


def test_exclude_drops_rows_seen_in_training(tmp_path):
    from distillops.cli import _exclude

    seen = tmp_path / "sft.jsonl"
    seen.write_text(json.dumps({"meta": {"id": EX.id}}) + "\n")
    assert _exclude(GSM8KTask(), [EX, EX_BIG], str(seen)) == [EX_BIG]
    assert _exclude(GSM8KTask(), [EX, EX_BIG], None) == [EX, EX_BIG]
