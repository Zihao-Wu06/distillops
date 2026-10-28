"""Build SFT datasets for the three data variants compared in the experiments.

    python scripts/rec_build_sft.py --variant noreason           # all train rows, no teacher
    python scripts/rec_build_sft.py --variant predict --src results/<cat>/teacher_predict_n0
    python scripts/rec_build_sft.py --variant oracle  --src results/<cat>/teacher_oracle_n0

`--match-rows` restricts the no-reasoning set to the same rows the teacher
accepted, so a "reasoning vs no reasoning" comparison uses identical users
(otherwise data size and user mix would confound the result).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from distillops.io import read_jsonl, write_json, write_jsonl  # noqa: E402
from tasks.rec.data import load_split  # noqa: E402
from tasks.rec.task import no_reasoning_rows  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default="Industrial_and_Scientific")
    ap.add_argument("--variant", choices=["noreason", "predict", "oracle"], required=True)
    ap.add_argument("--src", help="rejection-sampling output dir (for predict/oracle)")
    ap.add_argument("--match-rows", help="accepted.jsonl whose rows the no-reasoning set should match")
    ap.add_argument("--match-row-ids", help="text file with one train row number per line (same effect)")
    args = ap.parse_args()
    cat = args.category
    matched = args.match_rows or args.match_row_ids
    out = Path("data") / "sft" / cat / f"{args.variant}{'_matched' if matched else ''}.jsonl"

    if args.variant == "noreason":
        rows = load_split(cat, "train")
        if args.match_rows:
            keep = {r["meta"]["id"] for r in read_jsonl(args.match_rows)}
            rows = [r for r in rows if f"{cat}:train:{r.row}" in keep]
        elif args.match_row_ids:
            ids = {int(x) for x in Path(args.match_row_ids).read_text().split()}
            rows = [r for r in rows if r.row in ids]
        data = no_reasoning_rows(rows, cat)
    else:
        data = list(read_jsonl(Path(args.src) / "accepted.jsonl"))

    n = write_jsonl(out, data)
    write_json(out.with_suffix(".info.json"), {"variant": args.variant, "rows": n, "src": args.src})
    print(f"{n} rows -> {out}")


if __name__ == "__main__":
    main()
