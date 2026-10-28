"""Amazon benchmark data in the MiniOneRec format.

We reuse MiniOneRec's preprocessed Amazon 2018 Industrial / Office splits
(5-core, history capped at 10 items, chronological 8:1:1 split) so that our
numbers are directly comparable with the published table (SASRec, TIGER,
BIGRec, D3, S-DPO, MiniOneRec).

Files per category (from github.com/AkaliKong/MiniOneRec, Apache-2.0):
  {train,valid,test}/<cat>_5_2016-10-2018-11.csv   one row = (history -> next item)
  info/<cat>_5_2016-10-2018-11.txt                  catalog: "<sid>\t<title>\t<item_id>"
  index/<cat>.item.json                             item metadata (title, description, brand)
"""

from __future__ import annotations

import ast
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import httpx

RAW = "https://raw.githubusercontent.com/AkaliKong/MiniOneRec/main/data/Amazon"
SUFFIX = "_5_2016-10-2018-11"
CATEGORIES = ("Industrial_and_Scientific", "Office_Products")
HUMAN_NAME = {
    "Industrial_and_Scientific": "industrial and scientific items",
    "Office_Products": "office products",
}

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


@dataclass(frozen=True)
class Interaction:
    uid: str
    hist_titles: tuple[str, ...]
    hist_ids: tuple[int, ...]
    target_title: str
    target_id: int
    row: int  # row index inside its split file (stable example id)


@dataclass
class Catalog:
    titles: list[str]  # index = item_id
    sids: list[str]
    meta: dict[int, dict]

    @property
    def n_items(self) -> int:
        return len(self.titles)

    def unique_titles(self) -> list[str]:
        """Distinct titles; duplicates (different ASINs, same title) are
        indistinguishable for a title-generating model and for the metric."""
        return sorted(set(self.titles))


def _files(cat: str) -> dict[str, str]:
    return {
        "train": f"train/{cat}{SUFFIX}.csv",
        "valid": f"valid/{cat}{SUFFIX}.csv",
        "test": f"test/{cat}{SUFFIX}.csv",
        "info": f"info/{cat}{SUFFIX}.txt",
        "meta": f"index/{cat}.item.json",
    }


def download(cat: str, root: str | Path = "data/raw") -> Path:
    root = Path(root) / cat
    with httpx.Client(timeout=120, follow_redirects=True) as client:  # honours HTTPS_PROXY
        for rel in _files(cat).values():
            dst = root / rel
            if dst.exists() and dst.stat().st_size > 0:
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            r = client.get(f"{RAW}/{rel}")
            r.raise_for_status()
            dst.write_bytes(r.content)
    return root


def load_catalog(cat: str, root: str | Path = "data/raw") -> Catalog:
    base = Path(root) / cat
    rows = (base / _files(cat)["info"]).read_text(encoding="utf-8").splitlines()
    titles, sids = {}, {}
    for line in rows:
        parts = line.split("\t")
        sid, title, iid = parts[0].strip(), parts[1].strip(), int(parts[2])
        titles[iid], sids[iid] = title, sid
    n = max(titles) + 1
    assert sorted(titles) == list(range(n)), "item ids must be contiguous"
    meta_raw = json.loads((base / _files(cat)["meta"]).read_text(encoding="utf-8"))
    meta = {int(k): v for k, v in meta_raw.items()}
    return Catalog([titles[i] for i in range(n)], [sids[i] for i in range(n)], meta)


def load_split(cat: str, split: str, root: str | Path = "data/raw") -> list[Interaction]:
    path = Path(root) / cat / _files(cat)[split]
    out = []
    with open(path, encoding="utf-8", newline="") as f:
        for i, r in enumerate(csv.DictReader(f)):
            out.append(
                Interaction(
                    uid=r["user_id"],
                    hist_titles=tuple(t.strip() for t in ast.literal_eval(r["history_item_title"])),
                    hist_ids=tuple(int(x) for x in ast.literal_eval(r["history_item_id"])),
                    target_title=r["item_title"].strip(),
                    target_id=int(r["item_id"]),
                    row=i,
                )
            )
    return out
