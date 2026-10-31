"""Receipts as JSON: parsing model output, amounts, scoring and label-free checks.

A receipt is {"items": [{"name", "count", "price"}], "subtotal", "tax", "service", "discount",
"total", "cash", "change"}, every value a string as printed, or null. Scoring compares the
multiset of leaf values (item names, counts and prices, and each amount), as Donut's field F1
does: names case- and space-insensitive, numbers by value ("20,000" = "20.000" = "Rp 20000").
"""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

FIELDS = ("subtotal", "tax", "service", "discount", "total", "cash", "change")
ITEM_KEYS = ("name", "count", "price")


def _text(v: Any) -> str | None:
    if v is None:
        return None
    if isinstance(v, list):  # CORD stores a few values as lists
        v = " ".join(str(x) for x in v if x is not None)
    s = str(v).strip()
    return s or None


def parse_receipt(text: str) -> dict | None:
    """The receipt in a model's reply (bare JSON, a ```json block, or JSON inside other text),
    in canonical form; None if there is none."""
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    s = text.strip()
    try:
        data = json.loads(s)
    except json.JSONDecodeError:
        start, end = s.find("{"), s.rfind("}")
        if start == -1 or end <= start:
            return None
        try:
            data = json.loads(s[start: end + 1])
        except json.JSONDecodeError:
            return None
    if not isinstance(data, dict):
        return None
    items = data.get("items") or []
    if not isinstance(items, list) or not all(isinstance(i, dict) for i in items):
        return None
    return {"items": [{k: _text(i.get(k)) for k in ITEM_KEYS} for i in items],
            **{f: _text(data.get(f)) for f in FIELDS}}


def format_receipt(r: dict) -> str:
    """Canonical JSON for training targets."""
    return json.dumps({"items": [{k: i.get(k) for k in ITEM_KEYS} for i in r.get("items") or []],
                       **{f: r.get(f) for f in FIELDS}}, ensure_ascii=False)


def amount(v: Any) -> int | None:
    """A printed amount in hundredths ("20,000" and "20.000" -> 2000000, "12,50" -> 1250);
    None if it holds no digit. A final separator followed by 1-2 digits is a decimal point."""
    s = _text(v)
    if s is None:
        return None
    s = re.sub(r"[^\d.,]", "", s).strip(".,")
    if not re.search(r"\d", s):
        return None
    m = re.fullmatch(r"(.*?)[.,](\d{1,2})", s)
    whole, dec = (m.group(1), m.group(2)) if m else (s, "")
    return int(re.sub(r"\D", "", whole) or "0") * 100 + int(dec.ljust(2, "0") or "0")


def name_key(v: Any) -> str | None:
    s = _text(v)
    return " ".join(s.upper().split()) if s else None


def leaves(r: dict | None) -> Counter:
    """Multiset of (field, normalised value) pairs; missing values are left out."""
    out: Counter = Counter()
    for it in (r or {}).get("items") or []:
        for k in ITEM_KEYS:
            v = name_key(it.get(k)) if k == "name" else amount(it.get(k))
            if v is not None:
                out[(k, v)] += 1
    for f in FIELDS:
        v = amount((r or {}).get(f))
        if v is not None:
            out[(f, v)] += 1
    return out


def f1(pred: Counter, gold: Counter) -> float:
    if not pred and not gold:
        return 1.0
    return 2 * sum((pred & gold).values()) / (sum(pred.values()) + sum(gold.values()))


def _close(a: int, b: int) -> bool:
    """Equal up to cash rounding (1% or one unit)."""
    return abs(a - b) <= max(100, abs(b) // 100)


def rule_check(r: dict) -> tuple[bool, str]:
    """Label-free check of a receipt read by the teacher: the receipt's own arithmetic must hold
    wherever the values to check it are there (items add up to the subtotal, or to the total
    when their prices include tax; subtotal, tax, service and discount give the total; cash minus
    total is the change)."""
    items = r.get("items") or []
    total = amount(r.get("total"))
    if not items and total is None:
        return False, "empty"
    if any(not name_key(i.get("name")) for i in items):
        return False, "item_without_name"
    prices = [amount(i.get("price")) for i in items]
    known = items and all(p is not None for p in prices)
    items_sum = sum(prices) if known else None
    sub = amount(r.get("subtotal"))
    tax, svc, disc = (abs(amount(r.get(f)) or 0) for f in ("tax", "service", "discount"))

    def charged(base: int) -> set[int]:  # base with any of tax / service added and the discount taken off
        return {base + t * tax + s * svc - d * disc for t in (0, 1) for s in (0, 1) for d in (0, 1)}

    # item prices may include tax: their sum is then the subtotal plus tax, or the total itself
    if sub is not None and items_sum is not None and not any(
            _close(items_sum, x) for x in charged(sub) | ({total} if total is not None else set())):
        return False, "items_vs_subtotal"
    base = sub if sub is not None else items_sum
    if total is not None and base is not None and not any(_close(x, total) for x in charged(base)):
        return False, "total_mismatch"
    cash, change = amount(r.get("cash")), amount(r.get("change"))
    if cash is not None and change is not None and total is not None and not _close(cash - total, change):
        return False, "change_mismatch"
    return True, "ok"
