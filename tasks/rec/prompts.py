"""Prompt templates. The STUDENT prompt is defined once here and reused by SFT
export, evaluation and GRPO: train/inference template mismatch is the #1
silent bug in LLM fine-tuning."""

from __future__ import annotations

from typing import Sequence

THINK_OPEN, THINK_CLOSE = "<think>", "</think>"


def _history_block(titles: Sequence[str]) -> str:
    return "\n".join(f"{i + 1}. {t}" for i, t in enumerate(titles))


# ------------------------------------------------------------------ student --


def student_messages(hist_titles: Sequence[str], category_name: str) -> list[dict[str, str]]:
    return [
        {
            "role": "user",
            "content": (
                f"A customer bought these {category_name} in chronological order (oldest first):\n"
                f"{_history_block(hist_titles)}\n\n"
                "Predict the next product this customer will buy. "
                "Think about their needs first, then answer with the exact product title."
            ),
        }
    ]


def student_completion(target_title: str, reasoning: str | None) -> str:
    """Assistant text in Qwen3's thinking format. Empty think block = no-reasoning variant."""
    body = reasoning.strip() if reasoning else ""
    return f"{THINK_OPEN}\n{body}\n{THINK_CLOSE}\n\n{target_title}" if body else f"{THINK_OPEN}\n\n{THINK_CLOSE}\n\n{target_title}"


# ------------------------------------------------------------------ teacher --

TEACHER_SYSTEM = (
    "You are a senior e-commerce recommendation analyst. "
    "You infer a shopper's underlying needs from their purchase history."
)

ANALYSIS_RULES = (
    "Write a short analysis (60-120 words) covering: what the customer seems to be working on, "
    "brand / specification / compatibility patterns in the history, and what they most likely need next. "
    "Describe the need itself in plain words, as if no list of products had been shown. "
    "Do NOT mention candidate letters, list positions, 'the list', or the words "
    "'option', 'choice' or 'candidate'."
)


def teacher_predict_messages(
    hist_titles: Sequence[str], candidates: Sequence[str], letters: Sequence[str], category_name: str
) -> list[dict[str, str]]:
    cand = "\n".join(f"{l}. {t}" for l, t in zip(letters, candidates))
    return [
        {"role": "system", "content": TEACHER_SYSTEM},
        {
            "role": "user",
            "content": (
                f"A customer bought these {category_name} in chronological order (oldest first):\n"
                f"{_history_block(hist_titles)}\n\n"
                f"Candidate products (random order):\n{cand}\n\n"
                "Task: decide which ONE candidate the customer is most likely to buy next.\n"
                f"{ANALYSIS_RULES}\n"
                "Then, on the last line, write exactly: ANSWER: <letter>"
            ),
        },
    ]


def teacher_oracle_messages(
    hist_titles: Sequence[str], target_title: str, category_name: str
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": TEACHER_SYSTEM},
        {
            "role": "user",
            "content": (
                f"A customer bought these {category_name} in chronological order (oldest first):\n"
                f"{_history_block(hist_titles)}\n\n"
                f"Their next purchase was: {target_title}\n\n"
                "Write the reasoning a recommender should have done BEFORE knowing this purchase: "
                "infer the need from the history that leads to this kind of product. "
                f"{ANALYSIS_RULES} Write it as forward-looking prediction; never say 'they bought' "
                "or refer to knowing the answer.\n"
                "Then, on the last line, write exactly: ANSWER: done"
            ),
        },
    ]
