"""A toy VisionTask for the tests: name the colour of a solid-colour image."""

import json
import re
from pathlib import Path

from distillops.task import Verdict
from distillops_vision import VisionTask

COLOURS = {"red": (220, 30, 30), "green": (30, 200, 60), "blue": (40, 60, 220)}


def make_data(root: Path, per_split: int = 4) -> None:
    from PIL import Image

    root.mkdir(parents=True, exist_ok=True)
    for split, n in (("train", per_split), ("valid", 2), ("test", 3)):
        rows = []
        for i in range(n):
            name = list(COLOURS)[i % len(COLOURS)]
            path = root / f"{split}_{i}.png"
            Image.new("RGB", (60 + 10 * i, 40), COLOURS[name]).save(path)
            rows.append({"id": f"colours:{split}:{i}", "image": str(path), "colour": name})
        (root / f"{split}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")


class ColourTask(VisionTask):
    name = "colours"
    max_pixels = 40 * 40

    def __init__(self, data_dir: str = "data/colours") -> None:
        self.data_dir = Path(data_dir)

    def load_examples(self, split):
        path = self.data_dir / f"{split}.jsonl"
        if not path.exists():
            raise FileNotFoundError(path)
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    def example_id(self, ex):
        return ex["id"]

    def _messages(self, ex):
        return [{"role": "user", "content": [self.image(ex["image"]),
                                             {"type": "text", "text": "What colour is this image?"}]}]

    def teacher_messages(self, ex, attempt):
        return self._messages(ex)

    def parse_teacher(self, ex, text):
        found = re.findall(r"COLOUR:\s*(\w+)", text)
        return found[-1].lower() if found else None

    def verify(self, ex, parsed, text, attempt):
        if parsed is None:
            return Verdict(False, "parse_error")
        return Verdict(parsed == ex["colour"], "ok" if parsed == ex["colour"] else "wrong_colour")

    def sft_example(self, ex, parsed, text):
        return {"prompt": self.student_messages(ex), "completion": [{"role": "assistant", "content": f"COLOUR: {parsed}"}]}

    def student_messages(self, ex):
        return self._messages(ex)

    def score(self, ex, completion):
        return float(self.parse_teacher(ex, completion) == ex["colour"])

    def gold_sft_example(self, ex):
        return self.sft_example(ex, ex["colour"], "")


class MixedColourTask(ColourTask):
    """Every other prompt is text-only (a question about the colour's name, no image)."""

    name = "mixed_colours"

    def _messages(self, ex):
        i = int(ex["id"].rsplit(":", 1)[1])
        if i % 2:
            return [{"role": "user", "content": f"Which colour is called {ex['colour']}?"}]
        return super()._messages(ex)


class TwoImageTask(ColourTask):
    """Two images per prompt (the same picture twice)."""

    name = "two_images"

    def _messages(self, ex):
        return [{"role": "user", "content": [self.image(ex["image"]), self.image(ex["image"]),
                                             {"type": "text", "text": "What colour are these images?"}]}]
