"""Sanity-check a task before spending money: load data, show one teacher prompt,
one label row and how the student's chat template masks it.

    python scripts/check_task.py --task tasks.math.gsm8k:GSM8KTask --model Qwen/Qwen3-0.6B
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from distillops.cli import load_task  # noqa: E402
from distillops.sft import show_masked_example  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--task", required=True)
ap.add_argument("--task-arg", action="append", default=[])
ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
a = ap.parse_args()

from transformers import AutoTokenizer  # noqa: E402

task = load_task(a.task, a.task_arg)
for split in ("train", "test"):
    print(split, len(task.load_examples(split)), "examples")
ex = task.load_examples("train")[0]
print("--- teacher messages\n", task.teacher_messages(ex, 0))
row = task.gold_sft_example(ex)
print("--- label row, as the student sees it\n", show_masked_example(AutoTokenizer.from_pretrained(a.model), row))
