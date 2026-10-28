"""GRPO reward functions for recommendation (used in the RL stage).

The database of "what the user actually bought next" is the grader, so the
reward is verifiable. Two flavours, compared in the experiments:
- hit reward:  1 if the generated item is the true next item, else 0 (sparse);
- rank reward: for a ranked list (e.g. beam outputs), 1/log2(rank+2) of the
  true item, i.e. per-sample NDCG (denser: partial credit for "close" lists).
A small format penalty discourages broken think blocks.
"""

from __future__ import annotations

import math
from typing import Sequence

from .metrics import target_rank
from .prompts import THINK_CLOSE


def extract_title(completion: str) -> str:
    """Title = text after the closing think tag (or the whole text if absent)."""
    if THINK_CLOSE in completion:
        completion = completion.split(THINK_CLOSE, 1)[1]
    return completion.strip()


def hit_reward(completion: str, target: str) -> float:
    return 1.0 if extract_title(completion) == target.strip() else 0.0


def rank_reward(ranked: Sequence[str], target: str) -> float:
    r = target_rank(ranked, target)
    return 0.0 if r is None else 1.0 / math.log2(r + 2)


def format_penalty(completion: str, valid_titles: set[str], penalty: float = -0.1) -> float:
    if completion.count("<think>") != 1 or completion.count(THINK_CLOSE) != 1:
        return penalty
    return 0.0 if extract_title(completion) in valid_titles else penalty
