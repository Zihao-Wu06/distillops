"""Recommendation task for the distillation framework.

Two teacher modes (an ablation in the experiments):
- "predict": the teacher sees the history + 20 shuffled candidates (the true
  next item + 19 hard negatives from SASRec) and must pick one. Its reasoning
  is KEPT ONLY IF it picks the true item  ->  rejection sampling against
  ground truth.
- "oracle":  the teacher is told the true item and writes forward-looking
  reasoning for it (R2END-style rationalisation). Always on-target, but the
  reasoning is written with hindsight.

The student never sees candidates: it learns  history -> reasoning -> title.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from distillops.task import Task, Verdict

from .data import HUMAN_NAME, Interaction, load_catalog, load_split
from .prompts import student_completion, student_messages, teacher_oracle_messages, teacher_predict_messages
from .rewards import hit_reward

LETTERS = [chr(ord("A") + i) for i in range(26)]
_ANSWER = re.compile(r"ANSWER:\s*([A-Za-z]+)\s*$", re.MULTILINE)
# The student never sees a candidate list, so any reference to one is a leak.
_MENTIONS = re.compile(
    r"(?i:\b(?:option|candidate|choice)s?\b"
    r"|\bamong (?:the|these) (?:products|items|list)\b"
    r"|\b(?:listed|given|shown) (?:products|items)\b|\bthe list\b)"
    r"|\(\s*[A-T]\s*\)|\b(?:[Ii]tem|[Pp]roduct) [A-T]\b"
)
_HINDSIGHT = re.compile(r"\b(they|customer) (actually )?(bought|purchased)\b|\bnext purchase was\b|\bactually\b", re.IGNORECASE)


@dataclass(frozen=True)
class RecExample:
    inter: Interaction
    negatives: tuple[str, ...]  # hard-negative titles (target excluded)
    split: str = "train"


class RecTask(Task):
    def __init__(
        self,
        category: str = "Industrial_and_Scientific",
        mode: str = "predict",
        n_candidates: int = 20,
        min_words: int = 30,
        max_words: int = 180,
        seed: int = 0,
    ) -> None:
        assert mode in ("predict", "oracle")
        self.category = category
        self.category_name = HUMAN_NAME[category]
        self.mode = mode
        self.n_candidates = n_candidates
        self.min_words, self.max_words = min_words, max_words
        self.seed = seed
        self.name = f"rec-{mode}"

    # -------------------------------------------------------------- helpers --
    def candidates(self, ex: RecExample, attempt: int) -> tuple[list[str], int]:
        """Shuffled candidate list and the index of the true item. Reshuffled
        per attempt so a retry is not biased by the same position."""
        negs = list(dict.fromkeys(t for t in ex.negatives if t != ex.inter.target_title))
        cands = negs[: self.n_candidates - 1] + [ex.inter.target_title]
        rng = random.Random(f"{self.seed}-{ex.inter.row}-{attempt}")
        rng.shuffle(cands)
        return cands, cands.index(ex.inter.target_title)

    @staticmethod
    def split_answer(text: str) -> tuple[str, str | None]:
        matches = list(_ANSWER.finditer(text))
        if not matches:
            return text.strip(), None
        last = matches[-1]
        return text[: last.start()].strip(), last.group(1).strip()

    # ------------------------------------------------------------ interface --
    def load_examples(self, split: str) -> list[RecExample]:
        """Train rows carry SASRec hard negatives (from scripts/rec_prepare.py) for the
        teacher's candidate list; other splits need none."""
        from distillops.io import read_jsonl

        catalog = load_catalog(self.category)
        rows = load_split(self.category, split)
        path = Path("results") / self.category / "shortlist_train.jsonl"
        if split != "train" or not path.exists():  # negatives are only needed for teacher prompts
            return [RecExample(r, (), split) for r in rows]
        short = {r["row"]: r["ids"] for r in read_jsonl(path)}
        return [
            RecExample(r, tuple(catalog.titles[i] for i in short[r.row] if catalog.titles[i] != r.target_title), split)
            for r in rows
        ]

    def example_id(self, ex: RecExample) -> str:
        return f"{self.category}:{ex.split}:{ex.inter.row}"

    def teacher_messages(self, ex: RecExample, attempt: int) -> list[dict[str, str]]:
        if self.mode == "oracle":
            return teacher_oracle_messages(ex.inter.hist_titles, ex.inter.target_title, self.category_name)
        if not ex.negatives:
            raise ValueError("predict-mode teacher prompts need SASRec hard negatives: run scripts/rec_prepare.py "
                             "(results/<category>/shortlist_train.jsonl)")
        cands, _ = self.candidates(ex, attempt)
        return teacher_predict_messages(ex.inter.hist_titles, cands, LETTERS[: len(cands)], self.category_name)

    def parse_teacher(self, ex: RecExample, text: str) -> dict[str, Any] | None:
        analysis, answer = self.split_answer(text)
        if answer is None:
            return None
        return {"analysis": analysis, "answer": answer}

    def verify(self, ex: RecExample, parsed: dict | None, text: str, attempt: int = 0) -> Verdict:
        if parsed is None:
            return Verdict(False, "parse_error")
        words = len(parsed["analysis"].split())
        if words < self.min_words:
            return Verdict(False, "too_short", {"words": words})
        if words > self.max_words:
            return Verdict(False, "too_long", {"words": words})
        if _MENTIONS.search(parsed["analysis"]):
            return Verdict(False, "mentions_candidates")  # student will never see candidates
        if self.mode == "oracle":
            if _HINDSIGHT.search(parsed["analysis"]):
                return Verdict(False, "hindsight_leak")
            return Verdict(True, "ok", {"words": words})
        # predict mode: the core rejection-sampling check against ground truth
        letter = parsed["answer"].upper()
        cands, gold = self.candidates(ex, attempt)
        if letter not in LETTERS[: len(cands)]:
            return Verdict(False, "invalid_letter")
        picked = LETTERS.index(letter)
        if picked != gold:
            return Verdict(False, "wrong_item", {"picked": cands[picked]})
        return Verdict(True, "ok", {"words": words})

    def student_messages(self, ex: RecExample) -> list[dict[str, str]]:
        return student_messages(ex.inter.hist_titles, self.category_name)

    def score(self, ex: RecExample, completion: str) -> float:
        return hit_reward(completion, ex.inter.target_title)  # exact title after the think block

    def gold_sft_example(self, ex: RecExample) -> dict[str, Any]:
        return no_reasoning_rows([ex.inter], self.category)[0]

    def evaluate(self, model: str, examples: Sequence[RecExample], out_dir, **gen_kwargs: Any) -> dict[str, Any]:
        from .evaluate import evaluate_rec  # constrained beam search over the catalog

        return evaluate_rec(self, model, examples, out_dir, **gen_kwargs)

    def sft_example(self, ex: RecExample, parsed: dict, text: str) -> dict[str, Any]:
        return {
            "prompt": student_messages(ex.inter.hist_titles, self.category_name),
            "completion": [
                {"role": "assistant", "content": student_completion(ex.inter.target_title, parsed["analysis"])}
            ],
            "target_title": ex.inter.target_title,
        }


def no_reasoning_rows(rows: Sequence[Interaction], category: str) -> list[dict[str, Any]]:
    """Baseline SFT data without any teacher: history -> title (empty think block)."""
    name = HUMAN_NAME[category]
    return [
        {
            "prompt": student_messages(r.hist_titles, name),
            "completion": [{"role": "assistant", "content": student_completion(r.target_title, None)}],
            "target_title": r.target_title,
            "meta": {"id": f"{category}:train:{r.row}", "task": "rec-noreason"},
        }
        for r in rows
    ]
