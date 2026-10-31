"""VisionTask: a distillops Task whose prompts may contain images.

Write it like any Task (load_examples, example_id, teacher_messages, parse_teacher, verify,
sft_example, student_messages, score, optional gold_sft_example / extra_metrics); put images into
messages with `self.image(path)`:

    def student_messages(self, ex):
        return [{"role": "user", "content": [self.image(ex["image"]), {"type": "text", "text": "Total?"}]}]

`python -m distillops eval` and `python -m distillops labels` work on it unchanged (evaluation goes
through `evaluate` below); `teacher`, `sft`, `grpo` and `merge` come from `python -m distillops_vision`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from distillops.task import Task


class VisionTask(Task):
    # Largest image (width x height, in pixels). `image()` writes it into each image part, so the
    # teacher, the training rows, evaluation and GRPO all use the same picture: the teacher never
    # answers from detail the student cannot see. None = the image as stored (the student's
    # processor may still resize it to its own limits).
    max_pixels: int | None = None
    # A different size for the teacher only (None = the part's own). Prefer leaving it unset.
    teacher_max_pixels: int | None = None

    def image(self, path: str | Path) -> dict[str, Any]:
        part: dict[str, Any] = {"type": "image", "path": str(path)}
        if self.max_pixels:
            part["max_pixels"] = int(self.max_pixels)
        return part

    def evaluate(self, model: str, examples: Sequence[Any], out_dir: str | Path, **gen_kwargs: Any) -> dict[str, Any]:
        """The framework's evaluation (scoring, metrics.json, predictions.jsonl) with vision generation."""
        from distillops.evaluate import evaluate_task

        from .generate import generate_vision

        gen_kwargs.setdefault("max_pixels", self.max_pixels)
        return evaluate_task(self, model, examples, out_dir, generate_fn=generate_vision, **gen_kwargs)
