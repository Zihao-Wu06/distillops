"""Sanity-check a task before spending money: load its data, show one teacher prompt, the student
prompt, and (if the task has a label-only row) how the student's chat template masks a training row.

    python scripts/check_task.py --task tasks.math.gsm8k:GSM8KTask --model Qwen/Qwen3-0.6B
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from distillops.cli import load_task  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--task", required=True, help="module:Class or task YAML")
ap.add_argument("--task-arg", action="append", default=[], metavar="KEY=VALUE")
ap.add_argument("--model", required=True, help="the student (hub id or local path): its chat template is shown")
ap.add_argument("--chat-template", help="'plain', a Jinja file or string (default: the model's own, else plain)")
a = ap.parse_args()

from distillops.chat import load_tokenizer  # noqa: E402
from distillops.sft import show_masked_example  # noqa: E402

task = load_task(a.task, a.task_arg)
loaded = {}
for split in ("train", "valid", "test"):
    try:
        loaded[split] = task.load_examples(split)
        print(f"{split}: {len(loaded[split])} examples")
    except Exception as e:  # a task need not have every split
        print(f"{split}: not available ({type(e).__name__}: {e})")
examples = next((v for v in loaded.values() if v), None)
if not examples:
    raise SystemExit("no split returned any examples")
ex = examples[0]
print("--- example id:", task.example_id(ex))
print("--- teacher messages\n", task.teacher_messages(ex, 0))
tok = load_tokenizer(a.model, a.chat_template)
student = task.student_messages(ex)
print("--- student prompt, as the student sees it\n",
      tok.apply_chat_template(student, tokenize=False, add_generation_prompt=True))
try:
    row = task.gold_sft_example(ex)
except NotImplementedError:
    print("--- no gold_sft_example: the label-only control (`labels`) is not available for this task")
else:
    print("--- label row, as the student sees it\n", show_masked_example(tok, row))
