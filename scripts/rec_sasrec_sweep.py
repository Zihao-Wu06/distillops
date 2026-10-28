"""Small SASRec hyper-parameter sweep. Selection uses VALID only; the test set
is evaluated once, for the selected config (never tune on test).

    python scripts/rec_sasrec_sweep.py --category Industrial_and_Scientific
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from distillops.io import write_json, write_jsonl  # noqa: E402
from tasks.rec.baselines import SASRecConfig, SASRecRecommender  # noqa: E402
from tasks.rec.data import load_catalog, load_split  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default="Industrial_and_Scientific")
    ap.add_argument("--shortlist", type=int, default=50)
    args = ap.parse_args()
    cat = args.category
    out = Path("results") / cat
    catalog = load_catalog(cat)
    tr, va, te = (load_split(cat, s) for s in ("train", "valid", "test"))

    grid = {"hidden": [64, 128], "dropout": [0.3, 0.5], "weight_decay": [0.0, 1e-4], "tied": [False, True]}
    runs = []
    best = None
    for vals in itertools.product(*grid.values()):
        hp = dict(zip(grid, vals))
        rec = SASRecRecommender(catalog, SASRecConfig(**hp))
        fit = rec.fit(tr, va, log=lambda *_: None)
        runs.append({**hp, "valid_ndcg10": round(fit["best_valid_ndcg10"], 4), "epochs": fit["epochs_run"]})
        print(runs[-1], flush=True)
        if best is None or fit["best_valid_ndcg10"] > best[0]:
            best = (fit["best_valid_ndcg10"], hp, rec)

    _, hp, rec = best
    test = rec.evaluate(te)
    print("selected:", hp, "test:", test, flush=True)
    rec.save(str(out / "sasrec.pt"))
    top = rec.topk_ids(tr, args.shortlist)
    write_jsonl(out / "shortlist_train.jsonl", ({"row": r.row, "ids": [int(i) for i in t]} for r, t in zip(tr, top)))
    base = out / "baselines.json"
    import json

    results = json.loads(base.read_text(encoding="utf-8")) if base.exists() else {}
    results["sasrec_sweep"] = runs
    results["sasrec_selected"] = hp
    results["sasrec_test"] = test
    write_json(base, results)


if __name__ == "__main__":
    main()
