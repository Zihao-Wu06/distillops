from types import SimpleNamespace

import pytest

from distillops.io import read_jsonl
from distillops.rejection import rejection_sample
from distillops.task import Task, Verdict
from distillops.teacher import Price, Teacher, TeacherConfig, cached_prompt_tokens


class FakeClient:
    """Mimics openai.AsyncOpenAI: answer depends on the sample index in the prompt."""

    def __init__(self, answers):
        self.answers = answers  # function(messages, call_no) -> text
        self.calls = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, model, messages, extra_body=None, **params):
        self.calls += 1
        text = self.answers(messages, self.calls)
        usage = SimpleNamespace(prompt_tokens=1000, completion_tokens=100, prompt_cache_hit_tokens=800)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))], usage=usage)


def make_teacher(tmp_path, answers):
    cfg = TeacherConfig(
        base_url="x", model="m", cache_path=str(tmp_path / "c.sqlite"),
        price=Price(input_cache_hit=0.02, input_cache_miss=1.0, output=4.0),
    )
    return Teacher(cfg, client=FakeClient(answers))


class EvenTask(Task):
    """Toy task: the teacher must answer with the number in the prompt doubled."""

    name = "toy"

    def example_id(self, ex):
        return str(ex)

    def teacher_messages(self, ex, attempt):
        return [{"role": "user", "content": f"double {ex}"}]

    def parse_teacher(self, ex, text):
        return int(text) if text.strip().lstrip("-").isdigit() else None

    def verify(self, ex, parsed, text, attempt):
        if parsed is None:
            return Verdict(False, "parse_error")
        return Verdict(parsed == 2 * ex, "ok" if parsed == 2 * ex else "wrong")

    def sft_example(self, ex, parsed, text):
        return {"prompt": [{"role": "user", "content": str(ex)}], "completion": [{"role": "assistant", "content": text}]}


async def test_rejection_sampling_retries_and_resumes(tmp_path):
    # Teacher is wrong on its first call for every example, right afterwards.
    seen = {}

    def answers(messages, call_no):
        n = int(messages[0]["content"].split()[1])
        seen[n] = seen.get(n, 0) + 1
        return str(2 * n) if seen[n] > 1 else "garbage"

    teacher = make_teacher(tmp_path, answers)
    stats = await rejection_sample(EvenTask(), teacher, [1, 2, 3], tmp_path / "out", max_attempts=3)
    assert stats["yield"] == 1.0
    assert stats["attempts_until_accept"] == {2: 3}
    assert stats["rejection_reasons"] == {"parse_error": 3}
    assert len(list(read_jsonl(tmp_path / "out" / "accepted.jsonl"))) == 3
    # Re-running is free: finished examples are skipped (resume).
    calls_before = teacher.client.calls
    await rejection_sample(EvenTask(), teacher, [1, 2, 3], tmp_path / "out", max_attempts=3)
    assert teacher.client.calls == calls_before


async def test_teacher_disk_cache_and_cost(tmp_path):
    teacher = make_teacher(tmp_path, lambda m, c: "hello")
    a = await teacher.complete([{"role": "user", "content": "hi"}], sample_id=0)
    b = await teacher.complete([{"role": "user", "content": "hi"}], sample_id=0)
    c = await teacher.complete([{"role": "user", "content": "hi"}], sample_id=1)
    assert not a.cached and b.cached and not c.cached  # different sample_id -> new request
    assert teacher.client.calls == 2
    # cost: 800 hit*0.02 + 200 miss*1 + 100 out*4 per 1M tokens, the same at any time of day
    expected = (800 * 0.02 + 200 * 1.0 + 100 * 4.0) / 1e6
    assert abs(a.cost - expected) < 1e-12 and b.cost == 0.0  # the local cache is free
    summary = teacher.meter.summary()  # rounded for display; the meter keeps the exact sum
    assert summary["served_from_local_cache"] == 1 and abs(teacher.meter.cost - 2 * expected) < 1e-12
    assert summary["cost"] == round(2 * expected, 4)


async def test_stop_when_leaves_unfinished_examples_for_a_later_run(tmp_path):
    teacher = make_teacher(tmp_path, lambda messages, call_no: str(2 * int(messages[0]["content"].split()[1])))
    teacher.cfg.max_concurrency = 1  # examples run one at a time, so the stop point is deterministic
    calls = {"n": 0}

    def stop_after_two():  # e.g. a budget or deadline was reached
        calls["n"] += 1
        return calls["n"] > 2

    stats = await rejection_sample(EvenTask(), teacher, [1, 2, 3, 4], tmp_path / "out", stop_when=stop_after_two)
    assert stats["examples"] == 2 and stats["not_run_stopped"] == 2
    stats = await rejection_sample(EvenTask(), teacher, [1, 2, 3, 4], tmp_path / "out")  # resume finishes the rest
    assert stats["examples"] == 4 and stats["not_run_stopped"] == 0


def test_cached_prompt_tokens_for_different_providers():
    assert cached_prompt_tokens(SimpleNamespace(prompt_cache_hit_tokens=800)) == 800  # DeepSeek style
    assert cached_prompt_tokens(SimpleNamespace(prompt_tokens_details=SimpleNamespace(cached_tokens=300))) == 300  # OpenAI
    assert cached_prompt_tokens(SimpleNamespace(prompt_tokens_details={"cached_tokens": 5})) == 5
    assert cached_prompt_tokens(SimpleNamespace(prompt_tokens=10)) == 0  # provider without prompt caching
    assert cached_prompt_tokens(SimpleNamespace(prompt_tokens_details=None)) == 0


@pytest.mark.parametrize("path", ["configs/teacher_template.yaml", "configs/teacher_deepseek.yaml"])
def test_shipped_teacher_configs_load(path):
    cfg = TeacherConfig.from_yaml(path)
    assert cfg.base_url.startswith("https://") and cfg.model and cfg.api_key_env and cfg.cache_path


def test_teacher_config_defaults_are_provider_neutral():
    cfg = TeacherConfig(base_url="http://localhost:8000/v1", model="m")
    assert cfg.api_key_env == "TEACHER_API_KEY" and cfg.price == Price()


def test_teacher_command_requires_a_teacher_config():
    from distillops.cli import main

    with pytest.raises(SystemExit):  # argparse: --teacher is required
        main(["teacher", "--task", "tasks.math.gsm8k:GSM8KTask"])
