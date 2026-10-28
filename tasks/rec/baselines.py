"""Classic (non-LLM) baselines: Popularity and SASRec.

They give the floor and the "traditional recommender" reference row, and the
trained SASRec also produces candidate shortlists (hard negatives) for the
teacher prompts.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .data import Catalog, Interaction
from .metrics import ranking_metrics


def popularity_ranking(train: Sequence[Interaction], catalog: Catalog, depth: int = 50) -> list[str]:
    counts = Counter(x.target_id for x in train)
    top = [iid for iid, _ in counts.most_common(depth)]
    return [catalog.titles[i] for i in top]


def eval_popularity(train, test, catalog, depth: int = 50) -> dict[str, float]:
    ranked = popularity_ranking(train, catalog, depth)
    return ranking_metrics([ranked] * len(test), [x.target_title for x in test])


# ----------------------------------------------------------------- SASRec ---


@dataclass
class SASRecConfig:
    max_len: int = 10
    hidden: int = 64
    blocks: int = 2
    heads: int = 1
    dropout: float = 0.3
    lr: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 256
    epochs: int = 200
    patience: int = 10
    seed: int = 42
    tied: bool = False  # False = separate output layer (as MiniOneRec's SASRec)
    init_std: float = 0.02
    # LLM -> SASRec distillation (see scripts/rec_sasrec_distill.py)
    item_repr: str = "id"  # "id" | "fused" (ID + LLM title vector) | "llm" (LLM title vector only)
    llm_emb_path: str | None = None  # .npy [n_items, d] from tasks/rec/item_emb.py
    kd_weight: float = 0.0  # weight of the ranking-distillation loss (teacher top-K as soft labels)
    kd_topk: int = 0  # use only the teacher's first k items (0 = all); small k focuses on the head of the list
    kd_rank_power: float = 1.0  # sharpen rank weights (1/log2(rank+2))**power; >1 puts more mass on rank 1


def _build_model(n_items: int, cfg: SASRecConfig):
    import torch
    from torch import nn

    class SASRec(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.pad = n_items
            assert cfg.item_repr in ("id", "fused", "llm")
            if cfg.item_repr == "llm":
                assert cfg.tied, "item_repr='llm' needs tied scoring (an untied output layer is ID-based)"
            if cfg.item_repr in ("id", "fused"):
                self.item_emb = nn.Embedding(n_items + 1, cfg.hidden, padding_idx=n_items)
            if cfg.item_repr in ("fused", "llm"):
                emb = np.load(cfg.llm_emb_path).astype(np.float32)
                assert emb.shape[0] == n_items, f"{cfg.llm_emb_path}: {emb.shape[0]} rows for {n_items} items"
                emb = (emb - emb.mean(0)) / (emb.std(0) + 1e-6)  # standardise each dimension
                self.register_buffer("llm_emb", torch.from_numpy(emb))  # frozen: knowledge from the LLM
                # Trainable adapter into SASRec's space; init so its output matches the ID-embedding scale.
                self.adapter = nn.Linear(emb.shape[1], cfg.hidden)
                nn.init.normal_(self.adapter.weight, std=cfg.init_std / math.sqrt(emb.shape[1]))
                nn.init.zeros_(self.adapter.bias)
            self.pos_emb = nn.Embedding(cfg.max_len, cfg.hidden)
            self.drop = nn.Dropout(cfg.dropout)
            layer = nn.TransformerEncoderLayer(
                d_model=cfg.hidden,
                nhead=cfg.heads,
                dim_feedforward=cfg.hidden,
                dropout=cfg.dropout,
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, cfg.blocks, enable_nested_tensor=False)
            self.norm = nn.LayerNorm(cfg.hidden)
            self.out = None if cfg.tied else nn.Linear(cfg.hidden, n_items)
            # Small init: PyTorch's default N(0,1) embeddings give huge initial logits
            # (first-epoch loss ~32) and an over-confident, quickly over-fitting model.
            nn.init.normal_(self.pos_emb.weight, std=cfg.init_std)
            if hasattr(self, "item_emb"):
                nn.init.normal_(self.item_emb.weight, std=cfg.init_std)
                with torch.no_grad():
                    self.item_emb.weight[n_items].zero_()

        def item_vectors(self) -> "torch.Tensor":
            """[n_items + 1, hidden]; the last row (padding) is zero."""
            parts = []
            if hasattr(self, "item_emb"):
                parts.append(self.item_emb.weight)
            if hasattr(self, "llm_emb"):
                text = self.adapter(self.llm_emb)
                parts.append(torch.cat([text, text.new_zeros(1, text.size(1))]))
            return parts[0] if len(parts) == 1 else parts[0] + parts[1]

        def forward(self, seq: "torch.Tensor") -> "torch.Tensor":
            # seq: [B, L] right-aligned (padding on the left); the last position is the newest item.
            L = seq.size(1)
            items = self.item_vectors()
            pos = torch.arange(L, device=seq.device).unsqueeze(0)
            x = items[seq] * math.sqrt(cfg.hidden) + self.pos_emb(pos)
            x = self.drop(x)
            causal = torch.triu(torch.ones(L, L, dtype=torch.bool, device=seq.device), 1)
            pad_mask = seq.eq(self.pad)
            h = self.encoder(x, mask=causal, src_key_padding_mask=pad_mask)
            h = self.norm(h[:, -1])  # representation after the newest item
            if self.out is not None:
                return self.out(h)  # scores over all items
            return h @ items[: self.pad].T

    return SASRec()


def _soft_labels(topk: Sequence[Sequence[int]], n_items: int, k: int | None = None, power: float = 1.0):
    """Teacher top-K ids -> (ids [N, K] padded with -1, weights [N, K] summing to 1 per row).
    Rows without a teacher list get all-zero weights (no distillation signal)."""
    import torch

    k = k or max((len(t) for t in topk), default=1)
    ids = torch.full((len(topk), k), -1, dtype=torch.long)
    w = torch.zeros(len(topk), k)
    rank_w = torch.tensor([(1.0 / math.log2(r + 2)) ** power for r in range(k)])
    for i, t in enumerate(topk):
        t = [x for x in t if 0 <= x < n_items][:k]
        if t:
            ids[i, : len(t)] = torch.tensor(t)
            w[i, : len(t)] = rank_w[: len(t)] / rank_w[: len(t)].sum()
    return ids, w


def _to_tensor(rows: Sequence[Interaction], n_items: int, max_len: int):
    import torch

    seqs = np.full((len(rows), max_len), n_items, dtype=np.int64)
    for i, r in enumerate(rows):
        h = list(r.hist_ids)[-max_len:]
        seqs[i, max_len - len(h):] = h
    tgt = np.array([r.target_id for r in rows], dtype=np.int64)
    return torch.from_numpy(seqs), torch.from_numpy(tgt)


class SASRecRecommender:
    def __init__(self, catalog: Catalog, cfg: SASRecConfig | None = None, device: str | None = None) -> None:
        import torch

        self.catalog = catalog
        self.cfg = cfg or SASRecConfig()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(self.cfg.seed)
        random.seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        self.model = _build_model(catalog.n_items, self.cfg).to(self.device)

    def scores(self, rows: Sequence[Interaction], batch: int = 1024) -> np.ndarray:
        import torch

        self.model.eval()
        seq, _ = _to_tensor(rows, self.catalog.n_items, self.cfg.max_len)
        out = []
        with torch.no_grad():
            for i in range(0, len(rows), batch):
                out.append(self.model(seq[i : i + batch].to(self.device)).float().cpu().numpy())
        return np.concatenate(out)

    def topk_ids(self, rows: Sequence[Interaction], k: int = 50) -> np.ndarray:
        s = self.scores(rows)
        idx = np.argpartition(-s, k, axis=1)[:, :k]
        order = np.argsort(-np.take_along_axis(s, idx, 1), axis=1)
        return np.take_along_axis(idx, order, 1)

    def evaluate(self, rows: Sequence[Interaction], depth: int = 50) -> dict[str, float]:
        top = self.topk_ids(rows, depth)
        ranked = [[self.catalog.titles[i] for i in r] for r in top]
        return ranking_metrics(ranked, [r.target_title for r in rows])

    def fit(
        self,
        train: Sequence[Interaction],
        valid: Sequence[Interaction],
        log=print,
        teacher_topk: Sequence[Sequence[int]] | None = None,
    ) -> dict:
        """Full-softmax CE on the true next item, plus (if `teacher_topk` and cfg.kd_weight)
        ranking distillation: the teacher's top-K item ids for each training row become a
        soft label, weighted by rank like NDCG (1/log2(rank+2)), as in DLLM2Rec."""
        import torch
        import torch.nn.functional as F

        cfg = self.cfg
        seq, tgt = _to_tensor(train, self.catalog.n_items, cfg.max_len)
        soft = None
        if teacher_topk is not None and cfg.kd_weight > 0:
            soft = _soft_labels(teacher_topk, self.catalog.n_items, k=cfg.kd_topk or None, power=cfg.kd_rank_power)
        opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        best, best_state, bad, history = -1.0, None, 0, []
        for epoch in range(cfg.epochs):
            self.model.train()
            perm = torch.randperm(len(seq))
            total = 0.0
            for i in range(0, len(seq), cfg.batch_size):
                b = perm[i : i + cfg.batch_size]
                logits = self.model(seq[b].to(self.device))
                loss = F.cross_entropy(logits, tgt[b].to(self.device))  # full-softmax CE over the catalog
                if soft is not None:
                    ids, w = soft[0][b].to(self.device), soft[1][b].to(self.device)
                    logp = F.log_softmax(logits, dim=-1).gather(1, ids.clamp(min=0))
                    loss = loss + cfg.kd_weight * -(w * logp).sum(1).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item() * len(b)
            m = self.evaluate(valid)
            history.append({"epoch": epoch, "loss": round(total / len(seq), 4), **m})
            log(f"epoch {epoch:3d} loss {total / len(seq):.4f} valid NDCG@10 {m['NDCG@10']:.4f} HR@10 {m['HR@10']:.4f}")
            if m["NDCG@10"] > best:
                best, bad = m["NDCG@10"], 0
                best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
            else:
                bad += 1
                if bad >= cfg.patience:
                    break
        self.model.load_state_dict(best_state)
        return {"best_valid_ndcg10": best, "epochs_run": len(history), "history": history}

    def save(self, path: str) -> None:
        import torch

        torch.save({"cfg": self.cfg.__dict__, "state": self.model.state_dict()}, path)

    @classmethod
    def load(cls, path: str, catalog: Catalog, device: str | None = None) -> "SASRecRecommender":
        import torch

        ck = torch.load(path, map_location="cpu")
        rec = cls(catalog, SASRecConfig(**ck["cfg"]), device)
        rec.model.load_state_dict(ck["state"])
        return rec
