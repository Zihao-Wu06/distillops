"""Image parts, sizing, hashes, and the teacher: requests, cache keys, resumable runs."""

import asyncio
import base64
import io
import json
import shutil
import types

import pytest
from PIL import Image

from distillops_vision import images
from distillops_vision.teacher import VisionTeacher, VisionTeacherConfig, image_fingerprint


def test_fit_keeps_aspect_ratio_and_limit():
    assert images.fit((100, 50), None) == (100, 50)
    assert images.fit((100, 50), 10_000) == (100, 50)
    w, h = images.fit((400, 200), 10_000)
    assert w * h <= 10_000 and abs(w / h - 2) < 0.05


def test_parts_carry_their_own_size(toy_data):
    from vision_toy import ColourTask

    task = ColourTask()
    ex = task.load_examples("train")[0]
    msgs = task.student_messages(ex)
    (part,) = images.image_parts(msgs)
    assert part == {"type": "image", "path": ex["image"], "max_pixels": 1600}
    assert images.pixels(part, 999) == 1600 and images.pixels({"type": "image", "path": "x"}, 999) == 999
    im = images.load_part(part)
    assert im.size[0] * im.size[1] <= 1600 and im.mode == "RGB"
    templ = images.for_template(msgs + [{"role": "assistant", "content": "COLOUR: red"}])
    assert templ[0]["content"][0] == {"type": "image"}
    assert templ[1]["content"] == [{"type": "text", "text": "COLOUR: red"}]


def test_content_hash_follows_bytes_not_path(tmp_path):
    a = tmp_path / "a.png"
    Image.new("RGB", (8, 8), (1, 2, 3)).save(a)
    b = tmp_path / "moved" / "b.png"
    b.parent.mkdir()
    shutil.copy(a, b)
    assert images.content_hash(a) == images.content_hash(b)
    Image.new("RGB", (8, 8), (9, 9, 9)).save(b)
    assert images.content_hash(a) != images.content_hash(b)


def test_data_url_is_sized_and_typed(tmp_path):
    png, jpg = tmp_path / "x.png", tmp_path / "x.jpg"
    Image.new("RGB", (200, 100), (5, 5, 5)).save(png)
    Image.new("RGB", (200, 100), (5, 5, 5)).save(jpg)
    url = images.to_data_url(png, 5000)
    assert url.startswith("data:image/png;base64,")
    im = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))
    assert im.size[0] * im.size[1] <= 5000
    assert images.to_data_url(jpg).startswith("data:image/jpeg;base64,")


class FakeClient:
    """Answers with the colour it is told to; records every request."""

    def __init__(self, answer="COLOUR: red"):
        self.requests, self.answer = [], answer
        self.chat = types.SimpleNamespace(completions=self)

    async def create(self, **kw):
        self.requests.append(kw)
        msg = types.SimpleNamespace(content=self.answer)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)],
                                     usage=types.SimpleNamespace(prompt_tokens=100, completion_tokens=3))


def _teacher(tmp_path, client, **cfg):
    c = VisionTeacherConfig(base_url="http://x", model="vision-teacher", cache_path=str(tmp_path / "c.sqlite"), **cfg)
    return VisionTeacher(c, client=client)


def test_teacher_sends_images_as_image_url_at_the_parts_size(toy_data):
    from vision_toy import ColourTask

    task = ColourTask()
    ex = task.load_examples("train")[0]
    client = FakeClient()
    t = _teacher(toy_data, client)
    comp = asyncio.run(t.complete(task.teacher_messages(ex, 0)))
    assert comp.text == "COLOUR: red" and not comp.cached
    parts = client.requests[0]["messages"][0]["content"]
    assert parts[0]["type"] == "image_url" and parts[1] == {"type": "text", "text": "What colour is this image?"}
    sent = Image.open(io.BytesIO(base64.b64decode(parts[0]["image_url"]["url"].split(",", 1)[1])))
    assert sent.size[0] * sent.size[1] <= 1600  # the task's max_pixels, as the student sees it
    assert "path" not in json.dumps(client.requests[0]["messages"])  # no local paths leave the machine


def test_cache_key_follows_image_content_and_size(toy_data):
    from vision_toy import ColourTask

    task = ColourTask()
    ex = task.load_examples("train")[0]
    client = FakeClient()
    t = _teacher(toy_data, client)
    asyncio.run(t.complete(task.teacher_messages(ex, 0)))
    moved = toy_data / "elsewhere.png"
    shutil.copy(ex["image"], moved)
    again = asyncio.run(t.complete(task.teacher_messages({**ex, "image": str(moved)}, 0)))
    assert again.cached and len(client.requests) == 1  # same picture, new path: served from the cache
    Image.new("RGB", (60, 40), (0, 0, 0)).save(moved)
    asyncio.run(t.complete(task.teacher_messages({**ex, "image": str(moved)}, 0)))
    assert len(client.requests) == 2  # changed picture: asked again
    bigger = _teacher(toy_data, client, max_pixels=10_000)
    asyncio.run(bigger.complete(task.teacher_messages(ex, 0)))
    assert len(client.requests) == 3  # sent at another size: another request


def test_rejection_sampling_with_images_writes_rows_with_image_parts(toy_data):
    from distillops.rejection import rejection_sample
    from vision_toy import ColourTask

    task = ColourTask()
    examples = task.load_examples("train")
    stats = asyncio.run(rejection_sample(task, _teacher(toy_data, FakeClient()), examples, toy_data / "out", 2))
    assert stats["examples"] == 4 and 0 < stats["accepted"] < 4  # only the red ones pass
    rows = [json.loads(line) for line in (toy_data / "out" / "accepted.jsonl").read_text().splitlines()]
    (part,) = images.image_parts(rows[0]["prompt"])
    assert part["path"].endswith(".png") and part["max_pixels"] == 1600
    assert rows[0]["meta"]["id"].startswith("colours:train:")


def test_run_fingerprint_changes_with_the_images(toy_data):
    from vision_toy import ColourTask

    task = ColourTask()
    split = task.load_examples("train")
    before = image_fingerprint(task, split)
    Image.new("RGB", (60, 40), (1, 1, 1)).save(split[0]["image"])  # same path, other picture
    assert image_fingerprint(task, split) != before


def test_usage_counts_vision_teacher_calls(toy_data):
    from distillops.cli import main

    from vision_toy import ColourTask

    task = ColourTask()
    t = _teacher(toy_data, FakeClient())
    asyncio.run(t.complete(task.teacher_messages(task.load_examples("train")[0], 0)))
    cfg = toy_data / "t.yaml"
    cfg.write_text(f"base_url: http://x\nmodel: vision-teacher\ncache_path: {toy_data / 'c.sqlite'}\n")
    main(["usage", "--teacher", str(cfg)])  # the framework's own command reads the vision teacher's cache


@pytest.mark.parametrize("bad", [{"type": "image"}, {"type": "image", "url": "http://x"}])
def test_image_part_needs_a_path(bad):
    with pytest.raises(ValueError, match="path"):
        images.image_parts([{"role": "user", "content": [bad]}])
