"""Reading receipts (CORD-v2) as a distillops_vision task.

    python -m distillops_vision teacher --task vision_tasks.cord.task:CordTask --teacher <config>
    python -m distillops eval --task vision_tasks.cord.task:CordTask --split test --model <student>

Task arguments (--task-arg KEY=VALUE):
  verify    "rules" (default): the teacher's reading must pass the receipt's own arithmetic, no
            human label used; "gold": it must equal the annotation
  notes     1: add NOTES to the teacher's prompt (how these receipts are annotated, written from
            the teacher's answers on training receipts); the student's prompt never changes
  data_dir  folder written by vision_tasks/cord/prepare.py (default "data/cord")
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from distillops.task import Verdict
from distillops_vision import VisionTask

from .receipt import FIELDS, ITEM_KEYS, f1, format_receipt, leaves, parse_receipt, rule_check

PROMPT = """Read this receipt and return its contents as JSON with exactly these keys:
{"items": [{"name": ..., "count": ..., "price": ...}], "subtotal": ..., "tax": ..., "service": ..., "discount": ..., "total": ..., "cash": ..., "change": ...}
- items: every purchased item, in printed order. name: the item name as printed; count: the quantity; price: the amount printed on the item's line.
- subtotal, tax, service, discount: those amounts as printed; total: the amount to pay; cash: the amount paid in cash; change: the change given back.
- Copy names and numbers exactly as printed (keep separators, e.g. 20,000 or 20.000). Use null for anything the receipt does not show.
Answer with the JSON only."""
# Written after comparing the teacher's first answers with the labels of training receipts: it put
# the quantity and item codes into names, and listed add-ons and section headings as items.
NOTES = """

Notes on item names:
- name is the item's name only: leave out the quantity ("2 NASI GORENG" or "*2 PCS CHEESE TART" -> "NASI GORENG", "CHEESE TART"; the quantity goes in count) and an item code printed before the name ("05130000 ZUPPA SOUP" -> "ZUPPA SOUP").
- Options and add-ons printed under an item (e.g. "+DINGIN", "- SEDANG", "+B. TELOR BESAR") are not items: leave them out, and do not add them to the item's name.
- Section headings (e.g. "BEVERAGE", "FOOD") are not items."""
RETRY_HINT = "\n\nCheck again: the item prices, subtotal, total, cash and change must be read exactly."


class CordTask(VisionTask):
    name = "cord"
    max_pixels = 1024 * 1024  # every component sees the receipt at this size (about 1,000 image tokens)

    def __init__(self, verify: str = "rules", notes: int = 0, data_dir: str = "data/cord") -> None:
        if verify not in ("rules", "gold"):
            raise ValueError(f"verify must be 'rules' or 'gold' (got {verify!r})")
        self.verify_mode, self.notes, self.data_dir = verify, bool(int(notes)), Path(data_dir)

    def load_examples(self, split: str) -> list[dict]:
        path = self.data_dir / f"{split}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found: run vision_tasks/cord/prepare.py first")
        with open(path, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        for r in rows:  # image paths are stored relative to data_dir
            r["image"] = str(self.data_dir / r["image"])
        return rows

    def example_id(self, ex: dict) -> str:
        return ex["id"]

    def _messages(self, ex: dict, extra: str = "") -> list[dict]:
        return [{"role": "user", "content": [self.image(ex["image"]), {"type": "text", "text": PROMPT + extra}]}]

    def teacher_messages(self, ex: dict, attempt: int) -> list[dict]:
        return self._messages(ex, (NOTES if self.notes else "") + (RETRY_HINT if attempt else ""))

    def student_messages(self, ex: dict) -> list[dict]:
        return self._messages(ex)

    def parse_teacher(self, ex: dict, text: str) -> dict | None:
        return parse_receipt(text)

    def verify(self, ex: dict, parsed: Any, text: str, attempt: int) -> Verdict:
        if parsed is None:
            return Verdict(False, "parse_error")
        if self.verify_mode == "gold":
            ok = leaves(parsed) == leaves(ex["gold"])
            return Verdict(ok, "ok" if ok else "differs_from_gold")
        ok, reason = rule_check(parsed)
        return Verdict(ok, reason)

    def sft_example(self, ex: dict, parsed: Any, text: str) -> dict[str, Any]:
        return {"prompt": self.student_messages(ex),
                "completion": [{"role": "assistant", "content": format_receipt(parsed)}]}

    def gold_sft_example(self, ex: dict) -> dict[str, Any]:
        return {"prompt": self.student_messages(ex),
                "completion": [{"role": "assistant", "content": format_receipt(ex["gold"])}]}

    def score(self, ex: dict, completion: str) -> float:
        pred = parse_receipt(completion)
        return f1(leaves(pred), leaves(ex["gold"]))

    def extra_metrics(self, examples: Sequence[dict], completions: Sequence[str], scores: Sequence[float]) -> dict:
        """Micro precision / recall / F1 over all fields, F1 per field, the share of receipts read
        exactly, the share whose total is right, and the parse failure rate."""
        tp, fp, fn = Counter(), Counter(), Counter()
        exact = total_ok = parse_fail = 0
        for ex, c in zip(examples, completions):
            pred = parse_receipt(c)
            parse_fail += pred is None
            p, g = leaves(pred), leaves(ex["gold"])
            exact += p == g
            total_ok += {v for k, v in p if k == "total"} == {v for k, v in g if k == "total"}
            for (k, _), n in (p & g).items():
                tp[k] += n
            for (k, _), n in (p - g).items():
                fp[k] += n
            for (k, _), n in (g - p).items():
                fn[k] += n

        def prf(t: int, a: int, b: int) -> tuple[float, float, float]:
            prec, rec = (t / (t + a) if t + a else 0.0), (t / (t + b) if t + b else 0.0)
            return prec, rec, (2 * prec * rec / (prec + rec) if prec + rec else 0.0)

        n = max(len(examples), 1)
        p, r, micro = prf(sum(tp.values()), sum(fp.values()), sum(fn.values()))
        keys = ITEM_KEYS + FIELDS
        return {"precision": round(p, 4), "recall": round(r, 4), "micro_f1": round(micro, 4),
                "exact_match": round(exact / n, 4), "total_accuracy": round(total_ok / n, 4),
                "parse_fail_rate": round(parse_fail / n, 4),
                "f1_by_field": {k: round(prf(tp[k], fp[k], fn[k])[2], 4) for k in keys if tp[k] + fn[k]}}
