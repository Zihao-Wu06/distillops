"""How good is the label-free check? (uses the gold labels, for this report only)

1. Gold receipts that pass the arithmetic check (a check the human labels fail is too strict).
2. The teacher's first answers (eval_teacher on the same split with the labelling config: the
   same requests, served from the cache): F1 against gold for those the check accepts and rejects.

    python -m vision_tasks.cord.check_rules --split train [--predictions <eval_teacher dir>/predictions.jsonl]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter

from .receipt import f1, leaves, parse_receipt, rule_check
from .task import CordTask


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--data-dir", default="data/cord")
    ap.add_argument("--predictions")
    a = ap.parse_args()
    task = CordTask(data_dir=a.data_dir)
    exs = {e["id"]: e for e in task.load_examples(a.split)}
    reasons = Counter(rule_check(e["gold"])[1] for e in exs.values())
    print(f"gold receipts passing the check: {reasons['ok']}/{len(exs)}; failures: {dict(reasons - Counter(ok=reasons['ok']))}")
    if not a.predictions:
        return
    acc, rej = [], []
    with open(a.predictions, encoding="utf-8") as fh:
        for r in map(json.loads, fh):
            pred = parse_receipt(r["completion"])
            s = f1(leaves(pred), leaves(exs[r["id"]]["gold"]))
            (acc if pred is not None and rule_check(pred)[0] else rej).append(s)
    for name, xs in (("accepted", acc), ("rejected", rej)):
        if xs:
            print(f"first answers {name} by the check: {len(xs)}, mean F1 vs gold {sum(xs) / len(xs):.4f}, "
                  f"exact {sum(x == 1.0 for x in xs) / len(xs):.3f}")


if __name__ == "__main__":
    main()
