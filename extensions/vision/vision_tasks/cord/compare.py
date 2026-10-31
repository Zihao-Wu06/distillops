"""Paired comparison of two models' per-receipt scores (predictions.jsonl from `eval` or
eval_teacher): mean difference with a bootstrap 95% CI, and a two-sided sign test.

    python -m vision_tasks.cord.compare results/cord/eval/A/predictions.jsonl results/cord/eval/B/predictions.jsonl
"""

from __future__ import annotations

import json
import random
import sys
from math import comb


def scores(path: str) -> dict[str, float]:
    with open(path, encoding="utf-8") as f:
        return {r["id"]: float(r["score"]) for r in map(json.loads, f)}


def compare(a: dict[str, float], b: dict[str, float], reps: int = 10000, seed: int = 0) -> dict:
    ids = sorted(set(a) & set(b))
    d = [b[i] - a[i] for i in ids]
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(reps))
    up, down = sum(x > 0 for x in d), sum(x < 0 for x in d)
    n, k = up + down, min(up, down)
    p = min(1.0, 2 * sum(comb(n, j) for j in range(k + 1)) / 2 ** n) if n else 1.0
    return {"n": len(ids), "a": round(sum(a[i] for i in ids) / len(ids), 4), "b": round(sum(b[i] for i in ids) / len(ids), 4),
            "diff": round(sum(d) / len(d), 4), "ci95": [round(means[int(0.025 * reps)], 4), round(means[int(0.975 * reps)], 4)],
            "b_better": up, "b_worse": down, "sign_test_p": float(f"{p:.2g}")}


if __name__ == "__main__":
    print(json.dumps(compare(scores(sys.argv[1]), scores(sys.argv[2]))))
