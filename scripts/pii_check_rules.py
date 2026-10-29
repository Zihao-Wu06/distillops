"""How often the label-free rules reject the *gold* answer (they should almost never), with examples.

    python scripts/pii_check_rules.py --split train
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tasks.pii.entities import rule_check  # noqa: E402
from tasks.pii.task import PIITask  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--labels", default="coarse")
    ap.add_argument("--examples", type=int, default=3, help="examples to print per rejection reason")
    a = ap.parse_args()
    task = PIITask(labels=a.labels)
    rows = task.load_examples(a.split)
    reasons, shown = Counter(), Counter()
    for ex in rows:
        ok, reason = rule_check(ex["text"], task.gold(ex), task.cats)
        if ok:
            continue
        reasons[reason] += 1
        if shown[reason] < a.examples:
            shown[reason] += 1
            print(f"--- {reason}: {ex['id']}\n{ex['text'][:400]}\n gold: {task.gold(ex)}\n")
    bad = sum(reasons.values())
    print(f"{a.split}: gold answer rejected in {bad}/{len(rows)} texts ({bad / max(len(rows), 1):.2%}): {dict(reasons)}")


if __name__ == "__main__":
    main()
