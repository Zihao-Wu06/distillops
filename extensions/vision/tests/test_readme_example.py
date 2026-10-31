"""The task in README.md runs as written: teacher rows (fake teacher), labels, and its score."""

import asyncio
import json
import re
import types
from pathlib import Path

README = Path(__file__).parents[1] / "README.md"


def _example_source() -> str:
    blocks = re.findall(r"```python\n(.*?)```", README.read_text(encoding="utf-8"), re.S)
    (block,) = [b for b in blocks if "class ReceiptTotal" in b]
    return block


def test_readme_task_runs(tmp_path, monkeypatch):
    from PIL import Image

    from distillops.rejection import rejection_sample

    from distillops_vision import images
    from distillops_vision.teacher import VisionTeacher, VisionTeacherConfig

    monkeypatch.chdir(tmp_path)
    d = tmp_path / "data" / "receipts"
    d.mkdir(parents=True)
    rows = []
    for i, total in enumerate(["12.50", "3.00"]):
        Image.new("RGB", (1600, 900), (255, 255, 255)).save(d / f"{i}.png")
        rows.append({"id": f"r:{i}", "image": str(d / f"{i}.png"), "total": total})
    (d / "train.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    ns: dict = {}
    exec(_example_source(), ns)
    task = ns["ReceiptTotal"]()
    exs = task.load_examples("train")
    (part,) = images.image_parts(task.student_messages(exs[0]))
    assert part["max_pixels"] == 1024 * 1024 and images.load_part(part).size[0] * images.load_part(part).size[1] <= 1024 * 1024
    assert task.score(exs[0], "The total is TOTAL: 12.50") == 1.0 and task.score(exs[0], "TOTAL: 3.00") == 0.0

    class Client:
        def __init__(self):
            self.chat = types.SimpleNamespace(completions=self)

        async def create(self, **kw):
            msg = types.SimpleNamespace(content="TOTAL: 12.50")
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)],
                                         usage=types.SimpleNamespace(prompt_tokens=10, completion_tokens=3))

    cfg = VisionTeacherConfig(base_url="http://x", model="m", cache_path=str(tmp_path / "c.sqlite"))
    stats = asyncio.run(rejection_sample(task, VisionTeacher(cfg, client=Client()), exs, tmp_path / "out", 1))
    assert stats["accepted"] == 1  # only the receipt whose total is 12.50
    assert task.gold_sft_example(exs[1])["completion"][0]["content"] == "TOTAL: 3.00"
