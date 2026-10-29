import json

import pytest

from tasks.pii.task import PIITask


def _row(i, text, spans):
    return {"id": f"r{i}", "text": text, "region": "US",
            "spans": [[text.find(v), text.find(v) + len(v), label, v] for label, v in spans]}


ROWS = [
    _row(0, "Mr John Smith, born 4 May 1990, lives at 12 Oak Street, Boston 02108.",
         [("TITLE", "Mr"), ("GIVENNAME", "John"), ("SURNAME", "Smith"), ("DATE", "4 May 1990"),
          ("BUILDINGNUM", "12"), ("STREET", "Oak Street"), ("CITY", "Boston"), ("ZIPCODE", "02108")]),
    _row(1, "Email jane@corp.io, passport X1234567, age 34, female.",
         [("EMAIL", "jane@corp.io"), ("PASSPORTNUM", "X1234567"), ("AGE", "34"), ("GENDER", "female")]),
    _row(2, "The meeting is moved to the main hall.", []),
]


@pytest.fixture
def data(tmp_path):
    for split in ("train", "rl", "valid", "test"):
        (tmp_path / f"{split}.jsonl").write_text("\n".join(json.dumps(r) for r in ROWS) + "\n")
    return tmp_path


def test_gold_entities_in_both_granularities(data):
    coarse, fine = PIITask(data_dir=str(data)), PIITask(labels="fine", data_dir=str(data))
    ex = coarse.load_examples("train")[0]
    # coarse: adjacent same-category spans merge; "Oak Street, Boston 02108": the comma keeps Boston apart
    assert coarse.gold(ex) == [("NAME", "Mr John Smith"), ("DATE", "4 May 1990"), ("ADDRESS", "12 Oak Street"),
                               ("ADDRESS", "Boston 02108")]
    assert ("STREET", "Oak Street") in fine.gold(ex) and ("GIVENNAME", "John") in fine.gold(ex)
    assert "Dr Ana Maria Lee" in coarse.system_prompt() and "Dr Ana Maria Lee" not in fine.system_prompt()


def test_teacher_prompt_has_a_stable_prefix_and_worked_examples(data):
    task = PIITask(shots=2, data_dir=str(data))
    ex = task.load_examples("test")[2]
    first, retry = task.teacher_messages(ex, 0), task.teacher_messages(ex, 1)
    assert first[:-1] == retry[:-1] and first[-1] != retry[-1]  # only the last message changes on a retry
    assert len(first) == 1 + 2 * 2 + 1 and first[0]["role"] == "system"
    assert "NAME" in first[0]["content"] and "JSON" in first[0]["content"]
    # the worked examples are kept out of the RL pool
    shot_ids = {s["id"] for s in task.examples()}
    assert shot_ids and not shot_ids & {r["id"] for r in task.load_examples("rl")}
    assert task.student_messages(ex) == [first[0], {"role": "user", "content": f"Text:\n{ex['text']}"}]


def test_verify_rules_and_gold(data):
    rules, gold = PIITask(data_dir=str(data)), PIITask(verify="gold", data_dir=str(data))
    ex = rules.load_examples("train")[1]
    full = rules.gold(ex)
    assert rules.verify(ex, full, "", 0).accepted and gold.verify(ex, full, "", 0).accepted
    partial = [e for e in full if e[0] != "DEMOGRAPHIC"]
    assert rules.verify(ex, partial, "", 0).accepted  # rules cannot know an age was missed
    assert gold.verify(ex, partial, "", 0).reason == "differs_from_gold"
    no_email = [e for e in full if "@" not in e[1]]
    assert rules.verify(ex, no_email, "", 0).accepted  # no recall rule for emails (conventions differ)
    assert gold.verify(ex, no_email, "", 0).reason == "differs_from_gold"
    no_passport = [e for e in full if e[0] != "ID_NUMBER"]
    assert rules.verify(ex, no_passport, "", 0).accepted  # "X1234567" has only 7 digits: below the long-number rule
    assert rules.verify(ex, None, "", 0).reason == "parse_error"


def test_rows_score_and_metrics(data):
    task = PIITask(data_dir=str(data))
    exs = task.load_examples("test")
    rows = [task.gold_sft_example(ex) for ex in exs]
    assert rows[2]["completion"][0]["content"] == "[]"
    perfect = [r["completion"][0]["content"] for r in rows]
    assert [task.score(ex, c) for ex, c in zip(exs, perfect)] == [1.0, 1.0, 1.0]
    assert task.score(exs[0], "not json") == 0.0
    m = task.extra_metrics(exs, perfect, [1.0] * 3)
    assert m["micro_f1"] == m["macro_f1"] == m["value_f1"] == 1.0 and m["leak_rate"] == 0.0
    worse = ['[{"type": "NAME", "value": "John"}]', perfect[1], "oops"]
    m = task.extra_metrics(exs, worse, [0.0] * 3)
    # "John" alone is not the merged "Mr John Smith": 4 of 5 predictions are right
    assert m["recall"] < 1 and m["precision"] == 0.8 and m["parse_fail_rate"] == round(1 / 3, 4)
    # "John" alone does not match the merged "Mr John Smith"; leaks count the dataset's 12 original spans,
    # of which only "John" is masked
    assert m["leak_rate"] == round(7 / 12, 4)
    assert task.masked(exs[1], perfect[1]) == "Email [CONTACT], passport [ID_NUMBER], age [DEMOGRAPHIC], [DEMOGRAPHIC]."


def test_example_ids_are_text(data, tmp_path):
    # the dataset's uids are integers; ids must be text to match --keep-ids files and be joined into them
    row = dict(ROWS[0], id=20931646)
    (tmp_path / "int").mkdir()
    for split in ("train", "rl"):
        (tmp_path / "int" / f"{split}.jsonl").write_text(json.dumps(row) + "\n")
    task = PIITask(data_dir=str(tmp_path / "int"), shots=0)
    assert task.example_id(task.load_examples("train")[0]) == "20931646"


def test_bad_arguments_are_refused(data):
    with pytest.raises(ValueError):
        PIITask(verify="maybe", data_dir=str(data))
    with pytest.raises(ValueError):
        PIITask(labels="medium", data_dir=str(data))
    with pytest.raises(FileNotFoundError, match="prepare.py"):
        PIITask(data_dir=str(data / "missing")).load_examples("train")
