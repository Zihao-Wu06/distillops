"""Score the teacher itself on a split, with the same metrics as the student (one call per text,
with the teacher's prompt and worked examples). Responses are cached like every teacher call.

    python scripts/pii_eval_teacher.py --teacher configs/teacher_deepseek_pii.yaml --split test
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from distillops.cli import load_task, subsample  # noqa: E402
from distillops.io import write_json, write_jsonl  # noqa: E402
from distillops.teacher import Teacher, TeacherConfig  # noqa: E402


async def run(task, teacher, examples):
    async def one(ex):
        try:
            return (await teacher.complete(task.teacher_messages(ex, 0), sample_id=0)).text
        except Exception as e:  # counted as an unparseable answer (score 0)
            print(f"{task.example_id(ex)}: {type(e).__name__}: {e}", flush=True)
            return ""

    return await asyncio.gather(*(one(ex) for ex in examples))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--task", default="tasks.pii.task:PIITask")
    ap.add_argument("--task-arg", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", help="default: results/<task>/eval/teacher-<model>")
    a = ap.parse_args()

    cfg = TeacherConfig.from_yaml(a.teacher)
    task = load_task(a.task, a.task_arg)
    examples = subsample(task.load_examples(a.split), a.n, a.seed)
    teacher = Teacher(cfg)
    completions = asyncio.run(run(task, teacher, examples))
    scores = [task.score(ex, c) for ex, c in zip(examples, completions)]
    metrics = {"task": task.name, "model": f"teacher:{cfg.model}", "split": a.split, "n": len(examples),
               "score": round(sum(scores) / max(len(scores), 1), 4),
               **task.extra_metrics(examples, completions, scores), "teacher": teacher.meter.summary()}
    out = Path(a.out or f"results/{task.name}/eval/teacher-{cfg.model}")
    write_json(out / "metrics.json", metrics)
    write_jsonl(out / "predictions.jsonl", ({"id": task.example_id(ex), "score": s, "completion": c}
                                            for ex, s, c in zip(examples, scores, completions)))
    print(json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
