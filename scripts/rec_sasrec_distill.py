"""Distil an LLM recommender into SASRec (the model that would actually be deployed).

Two kinds of knowledge, both computed offline, so serving cost stays SASRec's:
  A. item understanding: LLM title embeddings -> SASRec item representation
  B. ranking knowledge:  the LLM's top-K list per training row -> soft labels (DLLM2Rec-style)

    python scripts/rec_sasrec_distill.py embed --model Qwen/Qwen3-0.6B --tag base06
    python scripts/rec_sasrec_distill.py embed --model checkpoints/rec_qwen3_0.6b_noreason/merged --tag sft06
    python scripts/rec_sasrec_distill.py topk  --model checkpoints/rec_qwen3_1.7b_noreason_matched/merged --tag sft17 \
        --beams 10 --n 12000 --exclude data/sft/Industrial_and_Scientific/noreason_matched.jsonl
    python scripts/rec_sasrec_distill.py train --emb sft06 --kd sft06 --seeds 3

Results are reported on the full test set and by how often the target item occurs in
training (0 = cold, 1-5 = rare, 6-20, >20 = head), plus the 1,000-row subset used for the LLMs.
Model selection (early stopping) uses the validation set only.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from distillops.io import append_jsonl, read_jsonl, write_json  # noqa: E402
from tasks.rec.data import load_catalog, load_split  # noqa: E402

GROUPS = [("cold (0)", 0, 0), ("rare (1-5)", 1, 5), ("mid (6-20)", 6, 20), ("head (>20)", 21, 10**9)]


def cmd_embed(a, cat: str) -> None:
    from tasks.rec.item_emb import title_embeddings

    catalog = load_catalog(cat)
    titles = [catalog.titles[i] for i in range(catalog.n_items)]
    emb = title_embeddings(a.model, titles)
    out = Path("results") / cat / "item_emb" / f"{a.tag}.npy"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, emb)
    print(f"{emb.shape} -> {out}")


def cmd_topk(a, cat: str) -> None:
    """Teacher top-K for every training row, written in chunks so a stopped run resumes."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from tasks.rec.data import HUMAN_NAME
    from tasks.rec.generate import build_trie, check_boundary, recommend_all

    catalog = load_catalog(cat)
    first_id = {}
    for i in range(catalog.n_items):
        first_id.setdefault(catalog.titles[i], i)  # duplicate titles: any of their ids is equivalent for SASRec
    rows = load_split(cat, "train")
    if a.exclude:  # rows the teacher was fine-tuned on: its lists there are memorised, not predictions
        seen = {r["meta"]["id"] for r in read_jsonl(a.exclude)}
        rows = [r for r in rows if f"{cat}:train:{r.row}" not in seen]
    if a.n:
        rows = random.Random(0).sample(rows, a.n)
    out = Path("results") / cat / "llm_topk" / f"{a.tag}.jsonl"
    done = {r["row"] for r in read_jsonl(out)} if out.exists() else set()
    todo = [r for r in rows if r.row not in done]
    print(f"{len(done)} rows done, {len(todo)} to go", flush=True)
    if not todo:
        return
    tok = AutoTokenizer.from_pretrained(a.model)
    from distillops.chat import device
    from distillops.precision import compute_dtype

    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=compute_dtype(), device_map=device()).eval()
    titles = catalog.unique_titles()
    check_boundary(tok, titles)
    trie = build_trie(titles, tok)
    for s in range(0, len(todo), a.chunk):
        part = todo[s : s + a.chunk]
        ranked, _ = recommend_all(model, tok, trie, [r.hist_titles for r in part], HUMAN_NAME[cat], think=False,
                                  num_beams=a.beams)
        for r, rk in zip(part, ranked):
            append_jsonl(out, {"row": r.row, "ids": [first_id[t] for t in rk if t in first_id]})
        print(f"teacher top-{a.beams}: {len(done) + s + len(part)}/{len(rows)}", flush=True)


def _eval(rec, rows, counts, catalog) -> dict:
    from tasks.rec.metrics import ranking_metrics

    top = rec.topk_ids(rows, 50)
    ranked = [[catalog.titles[i] for i in t] for t in top]
    res = {"all": ranking_metrics(ranked, [r.target_title for r in rows]), "n": len(rows)}
    for name, lo, hi in GROUPS:
        idx = [i for i, r in enumerate(rows) if lo <= counts[r.target_id] <= hi]
        if idx:
            res[name] = {**ranking_metrics([ranked[i] for i in idx], [rows[i].target_title for i in idx]), "n": len(idx)}
    return res


def cmd_train(a, cat: str) -> None:
    from tasks.rec.baselines import SASRecConfig, SASRecRecommender

    base_dir = Path("results") / cat
    catalog = load_catalog(cat)
    tr, va, te = (load_split(cat, s) for s in ("train", "valid", "test"))
    sub = random.Random(0).sample(te, 1000)  # the LLM evaluation subset
    counts = Counter()
    for r in tr:
        counts.update(list(r.hist_ids) + [r.target_id])
    selected = json.loads((base_dir / "baselines.json").read_text(encoding="utf-8"))["sasrec_selected"]
    emb_path = str(base_dir / "item_emb" / f"{a.emb}.npy") if a.emb else None
    teacher = None
    if a.kd:
        by_row = {r["row"]: r["ids"] for r in read_jsonl(base_dir / "llm_topk" / f"{a.kd}.jsonl")}
        teacher = [by_row.get(r.row, []) for r in tr]
        print(f"teacher lists for {sum(bool(t) for t in teacher)}/{len(tr)} training rows")

    variants = {
        "id": dict(item_repr="id"),
        "id-tied": dict(item_repr="id", tied=True),
    }
    if emb_path:
        variants |= {
            "fused": dict(item_repr="fused", llm_emb_path=emb_path),
            "fused-tied": dict(item_repr="fused", tied=True, llm_emb_path=emb_path),
            "llm-tied": dict(item_repr="llm", tied=True, llm_emb_path=emb_path),
        }
    if teacher:
        kd = {"kd_weight": a.kd_weight, "kd_topk": a.kd_topk, "kd_rank_power": a.kd_power}
        variants |= {f"{k}+kd": {**v, **kd} for k, v in list(variants.items()) if k in ("id", "fused-tied")}
    if a.only:
        variants = {k: v for k, v in variants.items() if k in a.only.split(",")}

    out = base_dir / "sasrec_distill" / (a.name or f"emb-{a.emb}_kd-{a.kd}")
    results = {}
    for name, kw in variants.items():
        runs = []
        for seed in range(42, 42 + a.seeds):
            cfg = SASRecConfig(**{**selected, **kw, "seed": seed})
            rec = SASRecRecommender(catalog, cfg)
            fit = rec.fit(tr, va, log=lambda *_: None, teacher_topk=teacher)
            runs.append({"seed": seed, "epochs": fit["epochs_run"], "valid_ndcg10": fit["best_valid_ndcg10"],
                         "test": _eval(rec, te, counts, catalog), "subset1000": _eval(rec, sub, counts, catalog)["all"]})
            t = runs[-1]["test"]
            print(f"{name:14s} seed {seed}: test HR@10 {t['all']['HR@10']:.4f} NDCG@10 {t['all']['NDCG@10']:.4f} | "
                  + " | ".join(f"{g} {t[g]['HR@10']:.3f}" for g, _, _ in GROUPS if g in t), flush=True)
        results[name] = {"config": kw, "runs": runs, "mean": _mean(runs)}
        write_json(out / "results.json", results)
    print(json.dumps({k: v["mean"] for k, v in results.items()}, indent=1))


def _mean(runs: list[dict]) -> dict:
    def stat(get):
        xs = [get(r) for r in runs]
        return {"mean": round(float(np.mean(xs)), 4), "std": round(float(np.std(xs)), 4)}

    out = {
        "test_HR@10": stat(lambda r: r["test"]["all"]["HR@10"]),
        "test_NDCG@10": stat(lambda r: r["test"]["all"]["NDCG@10"]),
        "subset1000_HR@10": stat(lambda r: r["subset1000"]["HR@10"]),
    }
    for g, _, _ in GROUPS:
        if all(g in r["test"] for r in runs):
            out[f"{g}_HR@10"] = stat(lambda r, g=g: r["test"][g]["HR@10"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default="Industrial_and_Scientific")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("embed")
    p.add_argument("--model", required=True)
    p.add_argument("--tag", required=True)
    p = sub.add_parser("topk")
    p.add_argument("--model", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--beams", type=int, default=20)
    p.add_argument("--n", type=int, default=0, help="random subset of training rows (0 = all)")
    p.add_argument("--chunk", type=int, default=2000)
    p.add_argument("--exclude", help="SFT jsonl whose rows the teacher was trained on (skipped)")
    p = sub.add_parser("train")
    p.add_argument("--emb", help="item_emb tag")
    p.add_argument("--kd", help="llm_topk tag")
    p.add_argument("--kd-weight", type=float, default=1.0)
    p.add_argument("--kd-topk", type=int, default=0, help="use only the teacher's first k items (0 = all)")
    p.add_argument("--kd-power", type=float, default=1.0, help="sharpen the rank weights")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--only", help="comma-separated variant names")
    p.add_argument("--name")
    a = ap.parse_args()
    {"embed": cmd_embed, "topk": cmd_topk, "train": cmd_train}[a.cmd](a, a.category)


if __name__ == "__main__":
    main()
