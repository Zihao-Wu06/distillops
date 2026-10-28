"""The custom-task example in README.md must work with the framework as written."""

import re
from pathlib import Path
from types import SimpleNamespace

from distillops.rejection import rejection_sample
from distillops.teacher import Price, Teacher, TeacherConfig

README = Path(__file__).resolve().parents[1] / "README.md"


def readme_task_class():
    section = README.read_text(encoding="utf-8").split("## Add your own task", 1)[1]
    code = re.search(r"```python\n(.*?)```", section, re.S).group(1)
    ns: dict = {}
    exec(compile(code, "README.md", "exec"), ns)
    return ns["TicketTriage"]


EXS = [{"id": "t1", "text": "I was charged twice", "label": "billing"},
       {"id": "t2", "text": "The app crashes on start", "label": "bug"}]


def test_readme_task_methods():
    task = readme_task_class()()
    ex = EXS[0]
    assert "LABEL:" in task.teacher_messages(ex, 0)[0]["content"]
    good = "Payment issue, so the billing team.\nLABEL: billing"
    assert task.verify(ex, task.parse_teacher(ex, good), good, 0).accepted
    wrong = "Sounds like a crash.\nLABEL: bug"
    assert task.verify(ex, task.parse_teacher(ex, wrong), wrong, 0).reason == "wrong_label"
    assert task.verify(ex, task.parse_teacher(ex, "no label"), "no label", 0).reason == "parse_error"
    row = task.sft_example(ex, "billing", good)
    assert row["prompt"] == task.student_messages(ex) and row["completion"][0]["content"] == good
    assert task.score(ex, good) == 1.0 and task.score(ex, wrong) == 0.0
    assert task.gold_sft_example(ex)["completion"][0]["content"] == "LABEL: billing"


class FakeClient:
    """Wrong label on the first call for each ticket, right on the retry."""

    def __init__(self):
        self.seen = {}
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, model, messages, extra_body=None, **params):
        text = messages[0]["content"]
        self.seen[text] = self.seen.get(text, 0) + 1
        label = "billing" if "charged" in text else "bug"
        answer = f"Reason.\nLABEL: {label if self.seen[text] > 1 else 'other'}"
        usage = SimpleNamespace(prompt_tokens=100, completion_tokens=20, prompt_cache_hit_tokens=0)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=answer))], usage=usage)


async def test_readme_task_runs_through_rejection_sampling(tmp_path):
    task = readme_task_class()()
    cfg = TeacherConfig(base_url="x", model="m", cache_path=str(tmp_path / "c.sqlite"), price=Price(output=1.0))
    stats = await rejection_sample(task, Teacher(cfg, client=FakeClient()), EXS, tmp_path / "out", max_attempts=3)
    assert stats["accepted"] == 2 and stats["attempts_until_accept"] == {2: 2}
    assert stats["rejection_reasons"] == {"wrong_label": 2}
