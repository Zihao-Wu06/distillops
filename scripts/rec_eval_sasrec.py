"""Evaluate the saved SASRec (and Popularity) on the same random test subset the
LLM evaluation uses (random.Random(0).sample(test, n)), so numbers are comparable.

    python scripts/rec_eval_sasrec.py --n 1000
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from distillops.io import write_json  # noqa: E402
from tasks.rec.baselines import SASRecRecommender, eval_popularity  # noqa: E402
from tasks.rec.data import load_catalog, load_split  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default="Industrial_and_Scientific")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    cat = args.category
    catalog = load_catalog(cat)
    train, test = load_split(cat, "train"), load_split(cat, "test")
    rows = random.Random(0).sample(test, args.n) if args.n else test
    rec = SASRecRecommender.load(str(Path("results") / cat / "sasrec.pt"), catalog, args.device)
    res = {"n": len(rows), "sasrec": rec.evaluate(rows), "popularity": eval_popularity(train, rows, catalog)}
    write_json(Path("results") / cat / f"classic_subset_n{args.n}.json", res)
    print(res)


if __name__ == "__main__":
    main()
