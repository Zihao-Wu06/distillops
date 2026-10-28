"""Evaluate an LLM recommender (base or fine-tuned) with constrained beam search.

    python scripts/rec_eval_llm.py --model Qwen/Qwen3-0.6B --n 200 --beams 20
    python scripts/rec_eval_llm.py --model Qwen/Qwen3-0.6B --n 200 --think
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from distillops.io import write_json, write_jsonl  # noqa: E402
from tasks.rec.data import HUMAN_NAME, load_catalog, load_split  # noqa: E402
from tasks.rec.generate import build_trie, check_boundary, recommend_all  # noqa: E402
from tasks.rec.metrics import ranking_metrics  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--category", default="Industrial_and_Scientific")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=0, help="random subset size (0 = full split)")
    ap.add_argument("--beams", type=int, default=20)
    ap.add_argument("--batch", type=int, default=0,
                    help="prompts per beam-search batch; 0 = auto from free GPU memory and the model config")
    ap.add_argument("--reason-batch", type=int, default=0, help="prompts per reasoning batch; 0 = auto")
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cat = args.category
    catalog = load_catalog(cat)
    rows = load_split(cat, args.split)
    if args.n:
        rows = random.Random(0).sample(rows, args.n)

    tok = AutoTokenizer.from_pretrained(args.model)
    from distillops.precision import compute_dtype

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=compute_dtype(), device_map="cuda")
    model.eval()
    titles = catalog.unique_titles()
    check_boundary(tok, titles)
    trie = build_trie(titles, tok)

    t0 = time.time()
    ranked_all, reasons_all = recommend_all(
        model, tok, trie, [r.hist_titles for r in rows], HUMAN_NAME[cat], args.think, num_beams=args.beams,
        beam_batch=args.batch, reason_batch=args.reason_batch,
    )
    print("example ranked[0][:3]:", ranked_all[0][:3], "| target:", rows[0].target_title)
    secs = time.time() - t0

    m = ranking_metrics(ranked_all, [r.target_title for r in rows])
    invalid = sum(t not in set(titles) for r in ranked_all for t in r)
    tag = args.tag or Path(args.model).name + ("-think" if args.think else "")
    out = Path("results") / cat / "llm_eval" / tag
    write_json(out / "metrics.json", {**m, "n": len(rows), "beams": args.beams, "invalid_items": invalid,
                                      "sec_per_example": round(secs / len(rows), 3), "model": args.model})
    write_jsonl(out / "predictions.jsonl", (
        {"row": r.row, "target": r.target_title, "ranked": rk, "reasoning": rs}
        for r, rk, rs in zip(rows, ranked_all, reasons_all)
    ))
    print(m, "| invalid items:", invalid, "| s/example:", round(secs / len(rows), 3))


if __name__ == "__main__":
    main()
