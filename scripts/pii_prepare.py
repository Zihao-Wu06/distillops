"""Build the English splits from OpenPII-1M (ai4privacy/pii-masking-openpii-1m, CC-BY-4.0).

    python scripts/pii_prepare.py --src /path/to/pii-masking-openpii-1m/data

Splits (all English, disjoint):
  train  texts the teacher labels (their gold labels are used only by the fully-labelled control)
  rl     small labelled pool for GRPO
  valid  GRPO validation (checkpoint selection)
  test   held-out evaluation, from the dataset's validation split
Each row: {"id", "text", "spans": [[start, end, label, value], ...], "region"}.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path


def read_english(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if '"language": "en"' not in line and '"language":"en"' not in line:
                continue  # cheap pre-filter; the parsed field is checked below
            r = json.loads(line)
            if r.get("language") != "en":
                continue
            spans = [[m["start"], m["end"], m["label"], m["value"]] for m in r["privacy_mask"]]
            rows.append({"id": str(r["uid"]), "text": r["source_text"], "spans": spans, "region": r.get("region")})
    return rows


def check(rows: list[dict], name: str) -> None:
    labels, bad = Counter(), 0
    for r in rows:
        for s, e, label, value in r["spans"]:
            labels[label] += 1
            bad += r["text"][s:e] != value
    lens = sorted(len(r["text"]) for r in rows)
    n_ent = sorted(len(r["spans"]) for r in rows)
    print(f"{name}: {len(rows)} rows | chars p50 {lens[len(lens) // 2]} max {lens[-1]} | "
          f"entities p50 {n_ent[len(n_ent) // 2]} max {n_ent[-1]} | spans whose offsets do not match their value: {bad}")
    print("  labels:", dict(labels.most_common()))


def write(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows -> {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="folder with the dataset's train.jsonl and validation.jsonl")
    ap.add_argument("--out", default="data/pii")
    ap.add_argument("--train", type=int, default=10_000)
    ap.add_argument("--rl", type=int, default=2_000)
    ap.add_argument("--valid", type=int, default=300)
    ap.add_argument("--test", type=int, default=2_000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    src, out = Path(a.src), Path(a.out)
    pool = read_english(src / "train.jsonl")
    held = read_english(src / "validation.jsonl")
    print(f"English rows: {len(pool)} in train, {len(held)} in validation")
    rng = random.Random(a.seed)
    rng.shuffle(pool)
    rng.shuffle(held)
    need = a.train + a.rl + a.valid
    if need > len(pool) or a.test > len(held):
        raise SystemExit(f"not enough English rows for the requested sizes ({need} train-side, {a.test} test)")
    splits = {"train": pool[: a.train], "rl": pool[a.train: a.train + a.rl],
              "valid": pool[a.train + a.rl: need], "test": held[: a.test]}
    for name, rows in splits.items():
        check(rows, name)
        write(rows, out / f"{name}.jsonl")


if __name__ == "__main__":
    main()
