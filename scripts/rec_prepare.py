"""Stage 1 for the recommendation task:
download data -> dataset stats -> Popularity + SASRec baselines -> SASRec
hard-negative shortlists for the teacher prompts.

    python scripts/rec_prepare.py --category Industrial_and_Scientific
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from distillops.io import write_json, write_jsonl  # noqa: E402
from tasks.rec.baselines import SASRecConfig, SASRecRecommender, eval_popularity  # noqa: E402
from tasks.rec.data import download, load_catalog, load_split  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default="Industrial_and_Scientific")
    ap.add_argument("--shortlist", type=int, default=50, help="SASRec top-N kept as hard negatives")
    args = ap.parse_args()
    cat = args.category
    out = Path("results") / cat
    out.mkdir(parents=True, exist_ok=True)

    download(cat)
    catalog = load_catalog(cat)
    splits = {s: load_split(cat, s) for s in ("train", "valid", "test")}
    stats = {
        "items": catalog.n_items,
        "unique_titles": len(catalog.unique_titles()),
        **{f"{s}_rows": len(v) for s, v in splits.items()},
        "train_users": len({r.uid for r in splits["train"]}),
        "avg_history_len": round(statistics.mean(len(r.hist_ids) for r in splits["train"]), 2),
    }
    print("dataset:", stats)

    results = {"dataset": stats, "popularity_test": eval_popularity(splits["train"], splits["test"], catalog)}
    print("popularity test:", results["popularity_test"])

    rec = SASRecRecommender(catalog, SASRecConfig())
    fit = rec.fit(splits["train"], splits["valid"])
    results["sasrec_fit"] = {k: v for k, v in fit.items() if k != "history"}
    results["sasrec_test"] = rec.evaluate(splits["test"])
    print("SASRec test:", results["sasrec_test"])
    rec.save(str(out / "sasrec.pt"))
    write_jsonl(out / "sasrec_history.jsonl", fit["history"])

    # Hard negatives for teacher prompts: SASRec's top-N for every TRAIN row.
    top = rec.topk_ids(splits["train"], args.shortlist)
    write_jsonl(
        out / "shortlist_train.jsonl",
        ({"row": r.row, "ids": [int(i) for i in t]} for r, t in zip(splits["train"], top)),
    )
    write_json(out / "baselines.json", results)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
