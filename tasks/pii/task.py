"""PII detection as a distillops task.

    python -m distillops teacher --task tasks.pii.task:PIITask --teacher configs/teacher_deepseek_pii.yaml
    python -m distillops eval    --task tasks.pii.task:PIITask --split test --model <student> --no-think

Task arguments (--task-arg KEY=VALUE):
  labels   "coarse" (7 categories, default) or "fine" (the dataset's 19 labels)
  verify   "rules" (default): the teacher's answer is checked without gold labels (known types,
           values copied verbatim, plausible formats, no email or long number left unmasked), as
           for a company's own unlabelled text; "gold": it must equal the annotation exactly
  shots    worked examples in the teacher's prompt, taken from the labelled `rl` pool (default 3)
  data_dir folder written by scripts/pii_prepare.py (default "data/pii")
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from distillops.task import Task, Verdict

from .entities import entity_set, f1, format_entities, leaked, mask_text, parse_entities, rule_check
from .labels import categories, to_category

HEAD = ("List every piece of personal information (PII) in the text so that it can be masked.\n"
        "- Copy each value exactly as it appears in the text, character for character.\n")
# coarse: one entry per run of same-category words (a full name, "12 Oak Street"); see PIITask.gold
SPANS = {"coarse": "- Words of the same category that stand next to each other form one entry: a full name with "
                   'its title ("Dr Ana Maria Lee"), a street with its number ("12 Oak Street"), a city with its postal '
                   'code ("Boston 02108"). Pieces separated by other words or by punctuation (a comma, a hyphen) are '
                   "separate entries.\n",
         "fine": "- List a title, the given name(s) and the surname(s) as separate entries; a given name or surname "
                 "of several words stays together. List street, building number, city and postal code "
                 "separately.\n"}
TAIL = ("- Every city or town name counts, even on its own. For an age, list only the number "
        '("27", not "27 years old"); gender and sex words count too.\n'
        "- List a value once even if it appears several times.\n"
        'Answer with a JSON list only: [{"type": "<CATEGORY>", "value": "<exact text>"}, ...], '
        "or [] if the text has no personal information.")
RETRY_HINT = "\n\nCheck again: include every personal detail and copy each value exactly as written."


class PIITask(Task):
    name = "pii"

    def __init__(self, labels: str = "coarse", verify: str = "rules", shots: int = 3, data_dir: str = "data/pii") -> None:
        if verify not in ("rules", "gold"):
            raise ValueError(f"verify must be 'rules' or 'gold' (got {verify!r})")
        self.cats = categories(labels)
        self.labels, self.verify_mode, self.shots, self.data_dir = labels, verify, shots, Path(data_dir)
        self._examples: list[dict] | None = None

    # ------------------------------------------------------------------ data --
    def _read(self, split: str) -> list[dict]:
        path = self.data_dir / f"{split}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found: run scripts/pii_prepare.py first (split {split!r})")
        with open(path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def load_examples(self, split: str) -> list[dict]:
        rows = self._read(split)
        if split == "rl":  # the teacher's worked examples come from this pool: keep them out of RL
            shot_ids = {ex["id"] for ex in self.examples()}
            rows = [r for r in rows if r["id"] not in shot_ids]
        return rows

    def example_id(self, ex: dict) -> str:
        return str(ex["id"])  # the dataset's uids are integers; ids are compared as text (e.g. --keep-ids files)

    def gold(self, ex: dict) -> list[tuple[str, str]]:
        """Gold entities. With coarse categories, adjacent spans of the same category separated only
        by whitespace are one entity ("Mr" + "John" + "Smith" -> "Mr John Smith"): where a given
        name ends and a surname starts is often unknowable from the text and does not change what
        is masked. Fine labels keep the dataset's own spans."""
        spans = sorted((s, e, to_category(label, self.labels)) for s, e, label, _ in ex["spans"])
        if self.labels == "fine":
            return [(kind, ex["text"][s:e]) for s, e, kind in spans]
        merged: list[list] = []
        for s, e, kind in spans:
            if merged and merged[-1][2] == kind and merged[-1][1] <= s and not ex["text"][merged[-1][1]:s].strip():
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e, kind])
        return [(kind, ex["text"][s:e]) for s, e, kind in merged]

    def examples(self) -> list[dict]:
        """Worked examples for the teacher: labelled `rl` rows chosen to cover many categories."""
        if self._examples is None:
            picked: list[dict] = []
            if self.shots > 0:
                pool = sorted(self._read("rl"), key=lambda r: r["id"])[:500]
                seen: set[str] = set()
                for _ in range(self.shots):
                    best = max((r for r in pool if r not in picked and len(r["text"]) <= 600),
                               key=lambda r: len({k for k, _ in self.gold(r)} - seen), default=None)
                    if best is None:
                        break
                    picked.append(best)
                    seen |= {k for k, _ in self.gold(best)}
            self._examples = picked
        return self._examples

    # --------------------------------------------------------------- prompts --
    def system_prompt(self) -> str:
        cats = "\n".join(f"- {name}: {desc}" for name, desc in self.cats.items())
        return f"{HEAD}{SPANS[self.labels]}{TAIL}\n\nCategories:\n{cats}"

    def teacher_messages(self, ex: dict, attempt: int) -> list[dict[str, str]]:
        msgs = [{"role": "system", "content": self.system_prompt()}]
        for shot in self.examples():  # same prefix for every request: served from the provider's prompt cache
            msgs += [{"role": "user", "content": f"Text:\n{shot['text']}"},
                     {"role": "assistant", "content": format_entities(self.gold(shot))}]
        msgs.append({"role": "user", "content": f"Text:\n{ex['text']}" + (RETRY_HINT if attempt else "")})
        return msgs

    def student_messages(self, ex: dict) -> list[dict[str, str]]:
        return [{"role": "system", "content": self.system_prompt()}, {"role": "user", "content": f"Text:\n{ex['text']}"}]

    # --------------------------------------------------------------- teacher --
    def parse_teacher(self, ex: dict, text: str) -> list[tuple[str, str]] | None:
        return parse_entities(text)

    def verify(self, ex: dict, parsed: Any, text: str, attempt: int) -> Verdict:
        if parsed is None:
            return Verdict(False, "parse_error")
        if self.verify_mode == "gold":
            ok = entity_set(parsed) == entity_set(self.gold(ex))
            return Verdict(ok, "ok" if ok else "differs_from_gold")
        ok, reason = rule_check(ex["text"], parsed, self.cats)
        return Verdict(ok, reason)

    def sft_example(self, ex: dict, parsed: Any, text: str) -> dict[str, Any]:
        return {"prompt": self.student_messages(ex),
                "completion": [{"role": "assistant", "content": format_entities(parsed)}]}

    def gold_sft_example(self, ex: dict) -> dict[str, Any]:
        return {"prompt": self.student_messages(ex),
                "completion": [{"role": "assistant", "content": format_entities(self.gold(ex))}]}

    # ------------------------------------------------------------ evaluation --
    def score(self, ex: dict, completion: str) -> float:
        pred = parse_entities(completion)
        return 0.0 if pred is None else f1(entity_set(pred), entity_set(self.gold(ex)))

    def extra_metrics(self, examples: Sequence[dict], completions: Sequence[str], scores: Sequence[float]) -> dict:
        """Corpus-level precision / recall / F1, per-category F1, type-agnostic F1, and the
        share of gold PII spans that would stay visible after masking (the leak rate)."""
        tp, fp, fn = Counter(), Counter(), Counter()
        vtp = vfp = vfn = spans = leaks = parse_fail = 0
        for ex, completion in zip(examples, completions):
            raw = parse_entities(completion)
            parse_fail += raw is None
            pred, gold = entity_set(raw or []), entity_set(self.gold(ex))
            for kind, _ in pred & gold:
                tp[kind] += 1
            for kind, _ in pred - gold:
                fp[kind] += 1
            for kind, _ in gold - pred:
                fn[kind] += 1
            pv, gv = {v for _, v in pred}, {v for _, v in gold}
            vtp, vfp, vfn = vtp + len(pv & gv), vfp + len(pv - gv), vfn + len(gv - pv)
            spans += len(ex["spans"])
            leaks += leaked(ex["text"], ex["spans"], [v for _, v in (raw or [])])

        def prf(t: int, p: int, n: int) -> tuple[float, float, float]:
            prec = t / (t + p) if t + p else 0.0
            rec = t / (t + n) if t + n else 0.0
            return prec, rec, (2 * prec * rec / (prec + rec) if prec + rec else 0.0)

        p, r, micro = prf(sum(tp.values()), sum(fp.values()), sum(fn.values()))
        per = {k: round(prf(tp[k], fp[k], fn[k])[2], 4) for k in self.cats}
        present = [per[k] for k in self.cats if tp[k] + fn[k]]  # categories that occur in the gold labels
        return {"precision": round(p, 4), "recall": round(r, 4), "micro_f1": round(micro, 4),
                "macro_f1": round(sum(present) / len(present), 4) if present else 0.0,
                "value_f1": round(prf(vtp, vfp, vfn)[2], 4),
                "leak_rate": round(leaks / spans, 4) if spans else 0.0,
                "parse_fail_rate": round(parse_fail / max(len(examples), 1), 4),
                "f1_by_category": per}

    def masked(self, ex: dict, completion: str) -> str:
        """The text as the student would mask it (for inspection)."""
        return mask_text(ex["text"], parse_entities(completion) or [])

