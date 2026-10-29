"""The task interface: everything task-specific lives behind these methods.

A new task (math, tool calling, extraction, recommendation, ...) subclasses
`Task`. The teacher client, rejection sampler, SFT, evaluation and GRPO are
shared, and the CLI (`python -m distillops ...`) drives any task by its import
path, e.g. `--task tasks.math.gsm8k:GSM8KTask`.

Required (teacher data):     load_examples, example_id, teacher_messages,
                             parse_teacher, verify, sft_example
Needed for eval / GRPO:      student_messages, score
Optional:                    gold_sft_example (label-only baseline),
                             evaluate (custom decoding, e.g. constrained beams)

One verifier, three uses: `verify` filters teacher data, `score` is both the
evaluation metric and the GRPO reward. Keeping them consistent is what makes
the pipeline measure what it trains for.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence


@dataclass
class Verdict:
    accepted: bool
    reason: str  # short machine-readable code, e.g. "ok", "wrong_answer", "parse_error"
    info: dict[str, Any] = field(default_factory=dict)


class Task(ABC):
    name: str = "task"

    # ---------------------------------------------------------------- data --
    def load_examples(self, split: str) -> list[Any]:
        """All examples of a split ("train", "test", ...). The CLI subsamples with --n."""
        raise NotImplementedError(f"{type(self).__name__} does not implement load_examples")

    @abstractmethod
    def example_id(self, ex: Any) -> str:
        """Stable id used for logging and resuming."""

    # ------------------------------------------------------------- teacher --
    @abstractmethod
    def teacher_messages(self, ex: Any, attempt: int) -> list[dict[str, str]]:
        """Chat messages sent to the teacher. `attempt` lets a task vary the prompt on retries."""

    @abstractmethod
    def parse_teacher(self, ex: Any, text: str) -> Any:
        """Parse the teacher's raw text into a structured answer (None if unparseable)."""

    @abstractmethod
    def verify(self, ex: Any, parsed: Any, text: str, attempt: int) -> Verdict:
        """Compare the teacher's answer with ground truth. This is the distillation filter.
        `attempt` is passed explicitly (never stored on self): examples run concurrently."""

    @abstractmethod
    def sft_example(self, ex: Any, parsed: Any, text: str) -> dict[str, Any]:
        """Turn an accepted teacher output into one student training row
        in TRL conversational prompt/completion format:
        {"prompt": [messages...], "completion": [{"role": "assistant", "content": ...}]}"""

    # ------------------------------------------------------------- student --
    def student_messages(self, ex: Any) -> list[dict[str, str]]:
        """The prompt the student sees (training rows and evaluation must use the same one)."""
        raise NotImplementedError(f"{type(self).__name__} does not implement student_messages")

    def score(self, ex: Any, completion: str) -> float:
        """Quality of a student completion in [0, 1]: evaluation metric and GRPO reward."""
        raise NotImplementedError(f"{type(self).__name__} does not implement score")

    def gold_sft_example(self, ex: Any) -> dict[str, Any]:
        """Optional label-only training row (no teacher). The control that answers
        "is the teacher's reasoning worth more than the plain labels?"."""
        raise NotImplementedError(f"{type(self).__name__} does not implement gold_sft_example")

    def extra_metrics(self, examples: Sequence[Any], completions: Sequence[str], scores: Sequence[float]) -> dict[str, Any]:
        """Optional metrics over the whole evaluation set, reported next to the mean `score`
        (e.g. macro-F1, BLEU, or anything that is not an average of per-example scores)."""
        return {}

    def evaluate(self, model: str, examples: Sequence[Any], out_dir: str | Path, **gen_kwargs: Any) -> dict[str, Any]:
        """Generate with the student and score. Override for custom decoding."""
        from .evaluate import evaluate_task

        return evaluate_task(self, model, examples, out_dir, **gen_kwargs)


def check_ids(task: Any, examples: Sequence[Any]) -> list[str]:
    """The examples' ids, after checking each is a non-empty str and all are distinct. Ids are
    compared as text with what earlier runs wrote (log.jsonl, --keep-ids, --exclude), so an id of
    another type would never match, and a repeated id would make two examples one."""
    ids: list[str] = []
    first: dict[str, int] = {}
    for i, ex in enumerate(examples):
        eid = task.example_id(ex)
        if not isinstance(eid, str) or not eid:
            raise TypeError(f"{type(task).__name__}.example_id must return a non-empty str, but example {i} gave "
                            f"{eid!r} ({type(eid).__name__}); e.g. return str(ex['id']).")
        if eid != eid.strip() or "\n" in eid or "\r" in eid:  # id files hold one id per line
            raise ValueError(f"{type(task).__name__}.example_id must not start or end with whitespace or contain "
                             f"a line break (example {i}: {eid!r}).")
        if eid in first:
            raise ValueError(f"{type(task).__name__}.example_id is not unique: examples {first[eid]} and {i} "
                             f"both have id {eid!r}.")
        first[eid] = i
        ids.append(eid)
    return ids
