"""Resuming only what the same prompt and code produced; failing fast on setup errors; text ids."""

import json
from types import SimpleNamespace

import pytest

from distillops.io import read_jsonl
from distillops.rejection import TeacherSetupError, check_run_settings, rejection_sample
from distillops.task import check_ids
from distillops.teacher import Teacher, TeacherConfig, is_request_error, is_transient
from tests.test_framework import EvenTask


class HTTPError(Exception):
    def __init__(self, status):
        super().__init__(f"HTTP {status}")
        self.status_code = status


class Client:
    """Fails with `errors` (one per call, None = answer correctly), then answers correctly."""

    def __init__(self, errors=()):
        self.errors, self.calls = list(errors), 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, model, messages, extra_body=None, **params):
        self.calls += 1
        if self.errors:
            err = self.errors.pop(0)
            if err is not None:
                raise err
        n = int(messages[0]["content"].split()[1])
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=str(2 * n)))],
                               usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2))


def teacher(tmp_path, client, **kw):
    cfg = TeacherConfig(base_url="x", model="m", cache_path=str(tmp_path / "c.sqlite"), **kw)
    return Teacher(cfg, client=client)


@pytest.fixture
def no_sleep(monkeypatch):
    async def instant(_):
        return None

    monkeypatch.setattr("distillops.teacher.asyncio.sleep", instant)


# -- 2: only network-type errors are retried; setup errors stop the run --------------------------

def test_error_classes():
    import httpx
    import openai

    for status in (408, 409, 429, 500, 502, 503):
        assert is_transient(HTTPError(status)) and is_request_error(HTTPError(status))
    for status in (400, 413, 422):
        assert not is_transient(HTTPError(status)) and is_request_error(HTTPError(status))  # this request only
    for status in (401, 403, 404):
        assert not is_transient(HTTPError(status)) and not is_request_error(HTTPError(status))  # every request
    conn = openai.APIConnectionError(request=httpx.Request("POST", "http://x"))
    for e in (conn, openai.APITimeoutError(request=httpx.Request("POST", "http://x")), ConnectionResetError(), TimeoutError()):
        assert is_transient(e) and is_request_error(e)
    for e in (TypeError("process() takes no keyword arguments"), ValueError("x"), KeyError("choices")):
        assert not is_transient(e) and not is_request_error(e)


async def test_local_errors_are_not_retried(tmp_path, no_sleep):
    client = Client([TypeError("process() takes no keyword arguments")])
    with pytest.raises(TypeError):
        await teacher(tmp_path, client).complete([{"role": "user", "content": "double 1"}])
    assert client.calls == 1


async def test_server_errors_are_retried(tmp_path, no_sleep):
    client = Client([HTTPError(503), HTTPError(429), None])
    out = await teacher(tmp_path, client).complete([{"role": "user", "content": "double 2"}])
    assert out.text == "4" and client.calls == 3


@pytest.mark.parametrize("err, hint", [(TypeError("process() takes no keyword arguments"), "installed packages"),
                                       (HTTPError(401), "API key")])
async def test_setup_errors_stop_the_run_and_keep_finished_examples(tmp_path, no_sleep, err, hint):
    t = teacher(tmp_path, Client([None, err]), max_concurrency=1)  # example 1 succeeds, then the error
    with pytest.raises(TeacherSetupError, match=hint):
        await rejection_sample(EvenTask(), t, [1, 2, 3], tmp_path / "out")
    # the example that hit the error and the one still waiting are not marked done
    assert len(list(read_jsonl(tmp_path / "out" / "log.jsonl"))) == 1
    assert t.client.calls == 2  # nothing was sent after the error
    t.client.errors = []  # fixed: a rerun finishes the rest
    stats = await rejection_sample(EvenTask(), t, [1, 2, 3], tmp_path / "out")
    assert stats["examples"] == 3 and stats["accepted"] == 3


async def test_an_error_about_one_request_is_recorded_and_the_run_goes_on(tmp_path, no_sleep):
    t = teacher(tmp_path, Client([HTTPError(400)]), max_concurrency=1)
    stats = await rejection_sample(EvenTask(), t, [1, 2], tmp_path / "out", max_attempts=1)
    assert stats["examples"] == 2 and stats["accepted"] == 1
    assert stats["rejection_reasons"] == {"api_error:HTTPError": 1}


# -- 3: ids are non-empty, unique strings; ids read from files are compared as text ---------------

class IntIdTask(EvenTask):
    def example_id(self, ex):
        return ex


def test_check_ids():
    assert check_ids(EvenTask(), [1, 2]) == ["1", "2"]
    with pytest.raises(TypeError, match="must return a non-empty str.*int"):
        check_ids(IntIdTask(), [1, 2])
    with pytest.raises(ValueError, match="not unique.*'1'"):
        check_ids(EvenTask(), [1, 2, 1])

    class Empty(EvenTask):
        def example_id(self, ex):
            return ""

    with pytest.raises(TypeError):
        check_ids(Empty(), [1])


async def test_int_ids_are_refused_before_any_call(tmp_path):
    t = teacher(tmp_path, Client())
    with pytest.raises(TypeError, match="example_id"):
        await rejection_sample(IntIdTask(), t, [1, 2], tmp_path / "out")
    assert t.client.calls == 0


async def test_logs_with_number_ids_still_resume(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "log.jsonl").write_text(json.dumps({"id": 1, "accepted": True, "attempts": [{"reason": "ok"}]}) + "\n")
    t = teacher(tmp_path, Client())
    stats = await rejection_sample(EvenTask(), t, [1, 2], out)  # an older run logged id 1 as a number
    assert t.client.calls == 1 and stats["examples"] == 2


def test_exclude_matches_number_ids_as_text(tmp_path):
    from distillops.cli import _exclude

    f = tmp_path / "rows.jsonl"
    f.write_text(json.dumps({"meta": {"id": 2}}) + "\n" + json.dumps({"meta": {}}) + "\n")
    assert _exclude(EvenTask(), [1, 2, 3], str(f)) == [1, 3]


def test_grpo_command_refuses_number_ids_before_loading_models(tmp_path, monkeypatch):
    from distillops import cli

    cfg = tmp_path / "g.yaml"
    cfg.write_text("model: m\noutput_dir: o\n")
    monkeypatch.setattr("tests.test_run_safety.IntIdTask.load_examples", lambda self, split: [1, 2], raising=False)
    with pytest.raises(TypeError, match="example_id"):
        cli.main(["grpo", str(cfg), "--task", "tests.test_run_safety:IntIdTask", "--keep-ids", str(tmp_path / "k")])


# -- 1: a resumed run must come from the same teacher prompt and task code ------------------------

def test_fingerprint_follows_the_prompt_not_the_subset():
    from distillops.cli import task_fingerprint

    a = task_fingerprint(EvenTask(), [3, 1, 2])
    assert a["teacher_prompt"] and a["task_code"]
    assert task_fingerprint(EvenTask(), [2, 1, 3]) == a  # same split, other order: same probe example

    class Reworded(EvenTask):
        def teacher_messages(self, ex, attempt):
            return [{"role": "user", "content": f"please double {ex}"}]

    b = task_fingerprint(Reworded(), [3, 1, 2])
    assert b["teacher_prompt"] != a["teacher_prompt"]
    assert task_fingerprint(EvenTask(), [])["teacher_prompt"] is None


def test_fingerprint_without_a_source_file():
    from distillops.cli import task_fingerprint

    ns: dict = {}
    exec("from tests.test_framework import EvenTask\nclass Live(EvenTask):\n    pass\n", ns)
    fp = task_fingerprint(ns["Live"](), [1])
    assert fp["teacher_prompt"]  # still compared
    # `exec` gives no file to read: task_code is None (the module of EvenTask is not this class's file)


def test_resume_refused_after_a_prompt_change(tmp_path, monkeypatch):
    from distillops import cli

    cfg = tmp_path / "t.yaml"
    cfg.write_text(f"base_url: http://x\nmodel: A\ncache_path: {tmp_path / 'c.sqlite'}\n")
    monkeypatch.setenv("TEACHER_API_KEY", "test")
    monkeypatch.setattr("tests.test_framework.EvenTask.load_examples", lambda self, split: [1, 2], raising=False)

    async def fake(task, teacher, examples, out, max_attempts):
        (out / "log.jsonl").write_text('{"id": "1"}\n')
        return {}

    monkeypatch.setattr("distillops.rejection.rejection_sample", fake)
    base = ["teacher", "--task", "tests.test_framework:EvenTask", "--teacher", str(cfg), "--out", str(tmp_path / "run")]
    cli.main(base)
    cli.main(base)  # same prompt and code: resumes
    monkeypatch.setattr("tests.test_framework.EvenTask.teacher_messages",
                        lambda self, ex, attempt: [{"role": "user", "content": f"twice {ex}"}])
    with pytest.raises(SystemExit, match="teacher_prompt"):
        cli.main(base)


def test_older_run_json_is_completed_not_refused(tmp_path, capsys):
    out = tmp_path / "run"
    out.mkdir()
    (out / "log.jsonl").write_text('{"id": "1"}\n')
    old = {"task": "m:T", "task_args": {}, "split": "train", "teacher": {"model": "A"}}
    (out / "run.json").write_text(json.dumps(old))
    check_run_settings(out, {**old, "teacher_prompt": "abc", "task_code": "def"})
    assert "does not record task_code, teacher_prompt" in capsys.readouterr().out
    assert json.loads((out / "run.json").read_text())["teacher_prompt"] == "abc"
    with pytest.raises(SystemExit, match="teacher_prompt"):  # recorded now, so a later change is caught
        check_run_settings(out, {**old, "teacher_prompt": "xyz", "task_code": "def"})


# -- replies that are not a chat completion (gateways answering 200 with an error body) -----------

def _real_client(handler):
    """The real openai.AsyncOpenAI over a mock transport (httpx2 for openai 3.x, httpx before)."""
    openai = pytest.importorskip("openai")
    try:
        import httpx2 as hx
    except ImportError:
        import httpx as hx
    http = openai.DefaultAsyncHttpxClient(transport=hx.MockTransport(handler))
    return hx, openai.AsyncOpenAI(base_url="http://x/v1", api_key="k", max_retries=0, http_client=http)


def _completion(n):
    return {"id": "x", "object": "chat.completion", "created": 0, "model": "m",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": str(2 * n)}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}


@pytest.mark.parametrize("bad", ["error_body", "not_json"])
async def test_bad_replies_are_retried_then_recorded_not_fatal(tmp_path, no_sleep, bad):
    replies = {"n": 0}

    def handler(req):
        replies["n"] += 1
        n = int(json.loads(req.content)["messages"][0]["content"].split()[1])
        if n == 2:  # example 2 always gets a bad reply; the others a normal one
            if bad == "error_body":
                return hx.Response(200, json={"error": {"message": "upstream overloaded", "code": 502}})
            return hx.Response(200, content=b"{not json", headers={"content-type": "application/json"})
        return hx.Response(200, json=_completion(n))

    hx, client = _real_client(handler)
    t = Teacher(TeacherConfig(base_url="x", model="m", cache_path=str(tmp_path / "c.sqlite"), max_retries=2),
                client=client)
    stats = await rejection_sample(EvenTask(), t, [1, 2, 3], tmp_path / "out", max_attempts=1)
    assert stats["examples"] == 3 and stats["accepted"] == 2  # the run went on
    assert stats["rejection_reasons"] == {"api_error:BadResponseError": 1}
    assert replies["n"] == 2 + 3  # examples 1 and 3 once, example 2 three times (two retries)


async def test_real_client_status_errors_are_classified(tmp_path, no_sleep):
    status = {"code": 401}
    hx, client = _real_client(lambda req: hx.Response(status["code"], json={"error": {"message": "no"}}))
    t = Teacher(TeacherConfig(base_url="x", model="m", cache_path=str(tmp_path / "c.sqlite")), client=client)
    with pytest.raises(TeacherSetupError, match="HTTP 401"):
        await rejection_sample(EvenTask(), t, [1], tmp_path / "a")
    status["code"] = 400
    stats = await rejection_sample(EvenTask(), t, [1], tmp_path / "b", max_attempts=1)
    assert stats["rejection_reasons"] == {"api_error:BadRequestError": 1}


def test_fingerprint_for_a_task_with_one_prompt_only():
    from distillops.cli import task_fingerprint

    class OnePrompt(EvenTask):
        def teacher_messages(self, ex, attempt):
            return [[{"role": "user", "content": f"double {ex}"}]][attempt]  # IndexError for attempt 1

    assert task_fingerprint(OnePrompt(), [1, 2])["teacher_prompt"]


def test_ids_with_outer_whitespace_or_line_breaks_are_refused():
    class Spaced(EvenTask):
        def example_id(self, ex):
            return {1: "q 1", 2: " q2", 3: "q\n3"}[ex]

    assert check_ids(Spaced(), [1]) == ["q 1"]  # inner spaces are fine: id files hold one id per line
    for bad in (2, 3):
        with pytest.raises(ValueError, match="whitespace"):
            check_ids(Spaced(), [bad])
