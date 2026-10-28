"""Two-stage recommendation with distilled SASRecs: recall, then re-rank.

Stage 1 (recall): the retriever (LLM item vectors + head-focused ranking distillation,
"version A'") scores the whole catalog; its top-N items are the candidates.
Stage 2 (re-rank): the candidates are re-ordered by
    alpha * log p_retriever + (1 - alpha) * log p_reranker
(alpha = 0: pure re-rank by the second model; alpha = 1: the retriever alone).
alpha is chosen on the validation set; the test set is only reported.

    python scripts/rec_two_stage.py --seed 42 --out results/.../two_stage/seed42.json
    python scripts/rec_two_stage.py --summarise results/.../two_stage
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from distillops.io import read_jsonl, write_json  # noqa: E402
from tasks.rec.data import load_catalog, load_split  # noqa: E402
from tasks.rec.metrics import ranking_metrics  # noqa: E402

CAT = "Industrial_and_Scientific"
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
GROUPS = [("cold (0)", 0, 0), ("rare (1-5)", 1, 5), ("mid (6-20)", 6, 20), ("head (>20)", 21, 10**9)]


def log_softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(1, keepdims=True)
    return x - np.log(np.exp(x).sum(1, keepdims=True))


def two_stage(ret: np.ndarray, rer: np.ndarray, alpha: float, n_cand: int) -> np.ndarray:
    """[rows, n_cand] item ids: retriever's top-n_cand, re-ordered by the blended score."""
    cand = np.argpartition(-ret, n_cand, axis=1)[:, :n_cand]
    blend = alpha * np.take_along_axis(log_softmax(ret), cand, 1) + (1 - alpha) * np.take_along_axis(log_softmax(rer), cand, 1)
    return np.take_along_axis(cand, np.argsort(-blend, axis=1), 1)


def metrics(ids: np.ndarray, rows, catalog, counts) -> dict:
    ranked = [[catalog.titles[i] for i in r] for r in ids]
    res = {"all": ranking_metrics(ranked, [r.target_title for r in rows])}
    for name, lo, hi in GROUPS:
        idx = [i for i, r in enumerate(rows) if lo <= counts[r.target_id] <= hi]
        res[name] = ranking_metrics([ranked[i] for i in idx], [rows[i].target_title for i in idx])
    return res


def run(seed: int, out: str, n_cand: int) -> None:
    from tasks.rec.baselines import SASRecConfig, SASRecRecommender

    base = Path("results") / CAT
    catalog = load_catalog(CAT)
    tr, va, te = (load_split(CAT, s) for s in ("train", "valid", "test"))
    counts = Counter()
    for r in tr:
        counts.update(list(r.hist_ids) + [r.target_id])
    hp = json.loads((base / "baselines.json").read_text(encoding="utf-8"))["sasrec_selected"]
    by_row = {r["row"]: r["ids"] for r in read_jsonl(base / "llm_topk" / "sft17.jsonl")}
    teacher = [by_row.get(r.row, []) for r in tr]
    emb = str(base / "item_emb" / "base17.npy")
    models = {  # the configurations chosen earlier on validation
        "retriever A'": dict(item_repr="fused", tied=True, llm_emb_path=emb, kd_weight=1.0, kd_topk=3),
        "reranker: original SASRec": dict(item_repr="id"),
        "reranker: SASRec + ranking KD": dict(item_repr="id", kd_weight=0.5),
    }
    scores = {}
    for name, kw in models.items():
        rec = SASRecRecommender(catalog, SASRecConfig(**{**hp, **kw, "seed": seed}))
        rec.fit(tr, va, log=lambda *_: None, teacher_topk=teacher)
        scores[name] = {"valid": rec.scores(va), "test": rec.scores(te)}
        print(f"seed {seed}: trained {name}", flush=True)

    ret = scores["retriever A'"]
    res = {"seed": seed, "n_candidates": n_cand, "retriever_only": metrics(two_stage(ret["test"], ret["test"], 1.0, n_cand), te, catalog, counts)}
    for name in [m for m in models if m.startswith("reranker")]:
        rer = scores[name]
        valid = {a: ranking_metrics([[catalog.titles[i] for i in r] for r in two_stage(ret["valid"], rer["valid"], a, n_cand)],
                                    [r.target_title for r in va])["NDCG@10"] for a in ALPHAS}
        best = max(valid, key=valid.get)
        res[name] = {"valid_ndcg10_by_alpha": valid, "alpha": best,
                     "test": metrics(two_stage(ret["test"], rer["test"], best, n_cand), te, catalog, counts),
                     "test_pure_rerank": metrics(two_stage(ret["test"], rer["test"], 0.0, n_cand), te, catalog, counts)}
        print(f"seed {seed}: {name}: valid NDCG@10 by alpha {valid} -> alpha {best}", flush=True)
    write_json(out, res)


def summarise(folder: str) -> None:
    runs = [json.loads(p.read_text()) for p in sorted(Path(folder).glob("seed*.json"))]
    keys = ["HR@3", "NDCG@3", "HR@5", "NDCG@5", "HR@10", "NDCG@10"]
    print(f"{len(runs)} seeds; mean over seeds (full test set)")
    print(f"{'pipeline':52s} " + " ".join(f"{k:>8s}" for k in keys) + "   cold    rare")

    def row(label, get):
        ms = [get(r) for r in runs]
        cells = " ".join(f"{st.mean(m['all'][k] for m in ms):8.4f}" for k in keys)
        cold = st.mean(m["cold (0)"]["HR@10"] for m in ms)
        rare = st.mean(m["rare (1-5)"]["HR@10"] for m in ms)
        print(f"{label:52s} {cells}  {cold:.4f}  {rare:.4f}")

    row("retriever A' alone", lambda r: r["retriever_only"])
    for name in [k for k in runs[0] if k.startswith("reranker")]:
        alphas = [r[name]["alpha"] for r in runs]
        row(f"A' top-{runs[0]['n_candidates']} -> {name.split(': ')[1]} (pure)", lambda r, n=name: r[n]["test_pure_rerank"])
        row(f"A' top-{runs[0]['n_candidates']} -> {name.split(': ')[1]} (alpha {alphas})", lambda r, n=name: r[n]["test"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--candidates", type=int, default=20)
    ap.add_argument("--out")
    ap.add_argument("--summarise")
    a = ap.parse_args()
    summarise(a.summarise) if a.summarise else run(a.seed, a.out, a.candidates)
