"""Paired comparison of two recommendation evals on the same users (McNemar-style sign test on HR@K).

    python scripts/rec_compare.py <eval_dir_a> <eval_dir_b> --k 10
"""

import argparse
import json
import math
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("a")
ap.add_argument("b")
ap.add_argument("--k", type=int, default=10)
a = ap.parse_args()


def load(d):
    rows = {}
    for l in open(Path(d) / "predictions.jsonl", encoding="utf-8"):
        r = json.loads(l)
        rows[r.get("row", r.get("id"))] = r["target"] in r["ranked"][: a.k]
    return rows


A, B = load(a.a), load(a.b)
keys = [k for k in A if k in B]
only_a = sum(A[k] and not B[k] for k in keys)
only_b = sum(B[k] and not A[k] for k in keys)
p = math.erfc(abs(only_a - only_b) / math.sqrt(2 * max(only_a + only_b, 1)))
print(f"n={len(keys)} HR@{a.k}: A={sum(A[k] for k in keys) / len(keys):.3f} B={sum(B[k] for k in keys) / len(keys):.3f} "
      f"| only A {only_a}, only B {only_b}, p~{p:.3f}")
