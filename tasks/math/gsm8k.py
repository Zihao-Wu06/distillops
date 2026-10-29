"""GSM8K grade-school math: the framework's second task (and the tutorial example).

Why it complements recommendation: the answer is a number, so verification is
exact and cheap, and multi-step reasoning is known to matter. Same pipeline,
opposite kind of task.

    python -m distillops teacher --task tasks.math.gsm8k:GSM8KTask --n 100
    python -m distillops labels  --task tasks.math.gsm8k:GSM8KTask                      # answer only
    python -m distillops labels  --task tasks.math.gsm8k:GSM8KTask --task-arg label_reasoning=human
    python -m distillops eval    --task tasks.math.gsm8k:GSM8KTask --model Qwen/Qwen3-0.6B --no-think
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from distillops.task import Task, Verdict

_ANSWER = re.compile(r"ANSWER:\s*(.+?)\s*$", re.MULTILINE)
_NUMBER = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?")
_CALC = re.compile(r"<<[^>]*>>")  # GSM8K calculator annotations, e.g. <<48/2=24>>

TEACHER_SYSTEM = "You are a careful math tutor who writes short, correct step-by-step solutions."
STUDENT_SUFFIX = "\n\nSolve step by step. Give the final answer on the last line as: ANSWER: <number>"


@dataclass(frozen=True)
class MathExample:
    id: str
    question: str
    solution: str  # human-written reference solution (GSM8K), used only by the label baselines
    answer: str  # gold final answer, normalised


def to_number(s: str) -> float | None:
    m = _NUMBER.findall(s)
    if not m:
        return None
    try:
        return float(m[0].replace("$", "").replace(",", ""))
    except ValueError:
        return None


def final_answer(text: str) -> float | None:
    """The student's/teacher's final answer: the last `ANSWER:` line, else the last number.
    Only the text after the reasoning block counts, so numbers inside <think> are ignored
    when an answer is given."""
    tail = text.rsplit("</think>", 1)[-1]
    found = _ANSWER.findall(tail)
    if found:
        return to_number(found[-1])
    nums = _NUMBER.findall(tail) or _NUMBER.findall(text)
    return to_number(nums[-1]) if nums else None


def same(a: float | None, b: float | None) -> bool:
    return a is not None and b is not None and abs(a - b) < 1e-6 * max(1.0, abs(b))


def _format_number(x: float) -> str:
    return str(int(x)) if float(x).is_integer() else f"{x:g}"


class GSM8KTask(Task):
    def __init__(self, max_words: int = 250, label_reasoning: str = "none", valid_size: int = 0) -> None:
        """valid_size > 0 sets the last `valid_size` training problems apart as split "valid" (and
        removes them from "train"), so GRPO can select checkpoints on problems SFT never saw."""
        assert label_reasoning in ("none", "human")
        assert valid_size >= 0
        self.max_words = max_words
        self.label_reasoning = label_reasoning
        self.valid_size = valid_size
        self.name = "gsm8k"

    # ---------------------------------------------------------------- data --
    def load_examples(self, split: str) -> list[MathExample]:
        from datasets import load_dataset

        source = "train" if split == "valid" and self.valid_size else split
        ds = load_dataset("openai/gsm8k", "main", split=source)
        out = []
        for i, r in enumerate(ds):
            solution, _, gold = r["answer"].rpartition("####")
            out.append(MathExample(f"gsm8k:{source}:{i}", r["question"], solution.strip(), gold.strip().replace(",", "")))
        if self.valid_size and source == "train":  # ids keep their train index, so logs and caches still match
            if self.valid_size >= len(out):
                raise ValueError(f"valid_size={self.valid_size} leaves no training problems")
            return out[-self.valid_size:] if split == "valid" else out[:-self.valid_size]
        return out

    def example_id(self, ex: MathExample) -> str:
        return ex.id

    # ------------------------------------------------------------- teacher --
    def teacher_messages(self, ex: MathExample, attempt: int) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": TEACHER_SYSTEM},
            {"role": "user", "content": (
                f"{ex.question}\n\nSolve it step by step in at most {self.max_words - 50} words: "
                "one short line per step, with the arithmetic written out. "
                "Then, on the last line, write exactly: ANSWER: <number>"
            )},
        ]

    def parse_teacher(self, ex: MathExample, text: str) -> dict[str, Any] | None:
        found = list(_ANSWER.finditer(text))
        if not found:
            return None
        value = to_number(found[-1].group(1))
        if value is None:
            return None
        return {"steps": text[: found[-1].start()].strip(), "value": value}

    def verify(self, ex: MathExample, parsed: dict | None, text: str, attempt: int = 0) -> Verdict:
        if parsed is None:
            return Verdict(False, "parse_error")
        words = len(parsed["steps"].split())
        if words == 0:
            return Verdict(False, "no_steps")
        if words > self.max_words:
            return Verdict(False, "too_long", {"words": words})
        if not same(parsed["value"], to_number(ex.answer)):
            return Verdict(False, "wrong_answer", {"got": parsed["value"]})
        return Verdict(True, "ok", {"words": words})

    def sft_example(self, ex: MathExample, parsed: dict, text: str) -> dict[str, Any]:
        return self._row(ex, parsed["steps"])

    # ------------------------------------------------------------- student --
    def student_messages(self, ex: MathExample) -> list[dict[str, str]]:
        return [{"role": "user", "content": ex.question + STUDENT_SUFFIX}]

    def score(self, ex: MathExample, completion: str) -> float:
        return 1.0 if same(final_answer(completion), to_number(ex.answer)) else 0.0

    def gold_sft_example(self, ex: MathExample) -> dict[str, Any]:
        steps = _CALC.sub("", ex.solution) if self.label_reasoning == "human" else ""
        return self._row(ex, steps)

    def _row(self, ex: MathExample, steps: str) -> dict[str, Any]:
        # Same layout as the recommendation task: reasoning inside the think block, answer after it.
        answer = _format_number(to_number(ex.answer))
        body = f"<think>\n{steps}\n</think>\n\nANSWER: {answer}" if steps else f"<think>\n\n</think>\n\nANSWER: {answer}"
        return {
            "prompt": self.student_messages(ex),
            "completion": [{"role": "assistant", "content": body}],
            "meta": {"id": ex.id, "task": self.name},
        }
