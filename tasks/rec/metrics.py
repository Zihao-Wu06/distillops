"""Ranking metrics, matching MiniOneRec's calc.py exactly.

A model returns a ranked list of item titles. The rank of the target is the
first position whose title equals the target title (string match). Then
  HR@K   = 1 if rank < K
  NDCG@K = 1 / log2(rank + 2) if rank < K   (single relevant item, ideal DCG = 1)
averaged over all test interactions.
"""

from __future__ import annotations

import math
from typing import Sequence

KS = (1, 3, 5, 10, 20, 50)


def target_rank(ranked: Sequence[str], target: str) -> int | None:
    target = target.strip()
    for i, t in enumerate(ranked):
        if t.strip() == target:
            return i
    return None


def ranking_metrics(
    ranked_lists: Sequence[Sequence[str]], targets: Sequence[str], ks: Sequence[int] = KS
) -> dict[str, float]:
    assert len(ranked_lists) == len(targets) and targets
    depth = min(len(r) for r in ranked_lists)
    ks = [k for k in ks if k <= depth]
    hr = {k: 0.0 for k in ks}
    ndcg = {k: 0.0 for k in ks}
    for ranked, tgt in zip(ranked_lists, targets):
        rank = target_rank(ranked, tgt)
        if rank is None:
            continue
        for k in ks:
            if rank < k:
                hr[k] += 1.0
                ndcg[k] += 1.0 / math.log2(rank + 2)
    n = len(targets)
    out = {}
    for k in ks:
        out[f"HR@{k}"] = round(hr[k] / n, 4)
        out[f"NDCG@{k}"] = round(ndcg[k] / n, 4)
    return out
