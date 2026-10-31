"""Write CORD-v2 as images + jsonl for CordTask.

    python vision_tasks/cord/prepare.py --src <snapshot of naver-clova-ix/cord-v2> --out data/cord

Splits: `train` (CORD train, 800 receipts) for the teacher; `test` (CORD validation + test, 200)
for evaluation only. Each row: {"id", "image" (relative to --out), "gold" (the receipt as JSON)}.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq


def _first(v):
    return v[0] if isinstance(v, list) and v and not isinstance(v[0], str) else v


def to_gold(gt: dict) -> dict:
    p = gt["gt_parse"]
    menu = p.get("menu") or []
    menu = [menu] if isinstance(menu, dict) else menu
    sub = _first(p.get("sub_total")) or {}
    tot = _first(p.get("total")) or {}
    keep = lambda v: (" ".join(map(str, v)) if isinstance(v, list) else v)  # noqa: E731
    return {"items": [{"name": keep(m.get("nm")), "count": keep(m.get("cnt")), "price": keep(m.get("price"))}
                      for m in menu if isinstance(m, dict)],
            "subtotal": keep(sub.get("subtotal_price")), "tax": keep(sub.get("tax_price")),
            "service": keep(sub.get("service_price")), "discount": keep(sub.get("discount_price")),
            "total": keep(tot.get("total_price")), "cash": keep(tot.get("cashprice")),
            "change": keep(tot.get("changeprice"))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", default="data/cord")
    a = ap.parse_args()
    out = Path(a.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    groups = {"train": ["train"], "test": ["validation", "test"]}
    for split, parts in groups.items():
        rows = []
        for part in parts:
            files = sorted(Path(a.src).glob(f"data/{part}-*.parquet"))
            i = 0
            for f in files:
                for rec in pq.read_table(f).to_pylist():
                    data = rec["image"]["bytes"]
                    ext = "png" if data[:4] == b"\x89PNG" else "jpg"
                    name = f"images/{part}_{i:04d}.{ext}"
                    (out / name).write_bytes(data)
                    rows.append({"id": f"cord:{part}:{i}", "image": name,
                                 "gold": to_gold(json.loads(rec["ground_truth"]))})
                    i += 1
        with open(out / f"{split}.jsonl", "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(split, len(rows), "receipts", flush=True)


if __name__ == "__main__":
    main()
