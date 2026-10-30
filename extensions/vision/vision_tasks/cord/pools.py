"""Human-labelled pools for GRPO, taken from the training receipts.

    python -m vision_tasks.cord.pools --accepted results/cord/teacher_train_notes/accepted.jsonl

Writes two splits next to train.jsonl:
- `valid` (--valid, default 100): receipts the student's SFT never saw (the teacher's reading
  failed the arithmetic check). Used only to pick the GRPO checkpoint. They are harder than
  average, which is fine for choosing between checkpoints.
- `rl` (--rl, default 200): a random sample of the other training receipts, scored against their
  human labels as the GRPO reward.
Human labels used by GRPO: rl + valid. The test receipts are never touched.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--accepted", required=True, help="the teacher run's accepted.jsonl (the SFT rows)")
    ap.add_argument("--data-dir", default="data/cord")
    ap.add_argument("--rl", type=int, default=200)
    ap.add_argument("--valid", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    data = Path(a.data_dir)
    rows = [json.loads(line) for line in open(data / "train.jsonl", encoding="utf-8") if line.strip()]
    with open(a.accepted, encoding="utf-8") as f:
        trained = {json.loads(line)["meta"]["id"] for line in f if line.strip()}
    unseen = [r for r in rows if r["id"] not in trained]
    seen = [r for r in rows if r["id"] in trained]
    rng = random.Random(a.seed)
    if len(unseen) < a.valid:
        raise SystemExit(f"only {len(unseen)} training receipts were not in the SFT data; lower --valid")
    valid = rng.sample(unseen, a.valid)
    rl = rng.sample(seen, min(a.rl, len(seen)))
    for name, part in (("valid", valid), ("rl", rl)):
        with open(data / f"{name}.jsonl", "w", encoding="utf-8") as fh:
            for r in sorted(part, key=lambda r: r["id"]):
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{name}: {len(part)} receipts", flush=True)


if __name__ == "__main__":
    main()
