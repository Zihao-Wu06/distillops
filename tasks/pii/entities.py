"""Entity lists: parsing model output, scoring, masking and label-free checks.

A prediction is a JSON list of {"type": <category>, "value": <exact text>}. Values, not character
offsets: models copy strings far more reliably than they count characters, and masking replaces
every occurrence of a value anyway. Entities are compared as a set of (type, value) pairs, so
listing a value twice or in another order does not matter.
"""

from __future__ import annotations

import json
import re
from typing import Iterable, Sequence

Entity = tuple[str, str]

LONG_NUMBER_RE = re.compile(r"\d(?:[\d \-./]*\d){8,}")  # 9+ digits, allowing spaces, dashes, dots and slashes
MONTHS = ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec")


def norm(value: str) -> str:
    """Whitespace-normalised value used for comparison (masking still uses the exact string)."""
    return " ".join(str(value).split())


def parse_entities(text: str) -> list[Entity] | None:
    """The entity list in a model's reply, or None if there is none. Accepts a bare JSON list,
    a ```json fenced block or a list surrounded by other text; a reasoning block ending in
    </think> is skipped."""
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    s = text.strip()
    data = None
    try:
        data = json.loads(s)
    except json.JSONDecodeError:
        start, end = s.find("["), s.rfind("]")
        if start != -1 and end > start:
            try:
                data = json.loads(s[start: end + 1])
            except json.JSONDecodeError:
                return None
    if not isinstance(data, list):
        return None
    out: list[Entity] = []
    for item in data:
        if not isinstance(item, dict):
            return None
        kind = item.get("type", item.get("label", item.get("category")))
        value = item.get("value")
        if not isinstance(kind, str) or not isinstance(value, str):
            return None
        out.append((kind.strip().upper(), value))
    return out


def format_entities(entities: Iterable[Entity]) -> str:
    """Canonical JSON for training targets: one entry per distinct (type, value), in order."""
    seen, rows = set(), []
    for kind, value in entities:
        key = (kind, norm(value))
        if key[1] and key not in seen:
            seen.add(key)
            rows.append({"type": kind, "value": value.strip()})
    return json.dumps(rows, ensure_ascii=False)


def entity_set(entities: Iterable[Entity]) -> set[Entity]:
    return {(kind, norm(value)) for kind, value in entities if norm(value)}


def f1(pred: set, gold: set) -> float:
    """F1 of two sets; 1.0 when both are empty (a text without PII, correctly left alone)."""
    if not pred and not gold:
        return 1.0
    return 2 * len(pred & gold) / (len(pred) + len(gold))


def covered(text: str, values: Iterable[str]) -> list[bool]:
    """Which characters of `text` fall inside an occurrence of one of `values`."""
    mask = [False] * len(text)
    for v in {v.strip() for v in values if v.strip()}:
        start = text.find(v)
        while start != -1:
            for i in range(start, start + len(v)):
                mask[i] = True
            start = text.find(v, start + 1)
    return mask


def leaked(text: str, spans: Sequence[Sequence], values: Iterable[str]) -> int:
    """How many gold spans [start, end, ...] would stay (partly) visible after masking `values`."""
    mask = covered(text, values)
    return sum(not all(mask[s:e]) for s, e, *_ in spans)


def mask_text(text: str, entities: Iterable[Entity]) -> str:
    """`text` with every occurrence of each value replaced by [TYPE]; longer values win overlaps."""
    owner: list[str | None] = [None] * len(text)
    for kind, value in sorted(set(entities), key=lambda e: -len(e[1].strip())):
        v = value.strip()
        start = text.find(v) if v else -1
        while start != -1:
            if all(o is None for o in owner[start: start + len(v)]):
                owner[start: start + len(v)] = [kind] * len(v)
            start = text.find(v, start + 1)
    out, i = [], 0
    while i < len(text):
        if owner[i] is None:
            out.append(text[i])
            i += 1
            continue
        kind, j = owner[i], i
        while j < len(text) and owner[j] == kind:
            j += 1
        out.append(f"[{kind}]")
        i = j
    return "".join(out)


def _digits(value: str) -> int:
    return sum(c.isdigit() for c in value)


def format_problem(kind: str, value: str) -> str | None:
    """A reason the value cannot be of this type, or None. Checks only what holds in any
    locale (an email has an @, a card number has at least 12 digits, ...)."""
    v = value.strip()
    if kind == "EMAIL" or (kind == "CONTACT" and "@" in v):
        local, _, domain = v.rpartition("@")  # real data has spaces and non-ASCII letters in addresses
        return None if local and "." in domain.strip(".") else "bad_email"
    if kind in ("TELEPHONENUM", "CONTACT"):
        return None if _digits(v) >= 7 else "bad_phone"
    if kind in ("CREDITCARDNUMBER", "CARD_NUMBER"):
        return None if _digits(v) >= 12 else "bad_card"
    if kind == "DATE":
        return None if _digits(v) or any(m in v.lower() for m in MONTHS) else "bad_date"
    return None


def rule_check(text: str, entities: Sequence[Entity], allowed: Iterable[str]) -> tuple[bool, str]:
    """Label-free verification of a teacher's answer: known types, values copied verbatim from
    the text, plausible formats, and no long number (9+ digits) left unmasked. Emails are not
    required to be masked: annotation conventions differ on company addresses (support@...)."""
    allowed = set(allowed)
    values = []
    for kind, value in entities:
        v = value.strip()
        if kind not in allowed:
            return False, "unknown_type"
        if not v or len(v) > 100:
            return False, "bad_value"
        if v not in text:
            return False, "value_not_in_text"
        problem = format_problem(kind, v)
        if problem:
            return False, problem
        values.append(v)
    mask = covered(text, values)
    for m in LONG_NUMBER_RE.finditer(text):  # every digit must be masked; separators may stay
        if not all(mask[i] for i in range(m.start(), m.end()) if text[i].isdigit()):
            return False, "missed_number"
    return True, "ok"
