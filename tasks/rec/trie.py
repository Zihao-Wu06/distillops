"""Constrained decoding over the item catalog.

Every catalog title is tokenized into a path of a prefix tree (trie). While
the model writes the title, at each step only tokens that continue some real
title are allowed; when a full title is complete, EOS becomes allowed. The
model therefore can never output an item that does not exist, and beam search
over the trie returns a ranked list of real items (= a top-K recommendation).
"""

from __future__ import annotations

from typing import Iterable, Sequence

_END = -1  # marker key: a complete title ends at this node


class TitleTrie:
    def __init__(self, token_seqs: Iterable[Sequence[int]], eos_id: int) -> None:
        self.root: dict = {}
        self.eos_id = eos_id
        self.size = 0
        self._memo: dict[tuple, list[int]] = {}
        self.max_depth = 0  # longest title in tokens: generation must allow this many steps (+EOS)
        for seq in token_seqs:
            self.max_depth = max(self.max_depth, len(seq))
            node = self.root
            for t in seq:
                node = node.setdefault(int(t), {})
            if _END not in node:
                node[_END] = True
                self.size += 1

    @classmethod
    def from_titles(cls, titles: Iterable[str], tokenizer) -> "TitleTrie":
        seqs = [tokenizer(t, add_special_tokens=False)["input_ids"] for t in dict.fromkeys(titles)]
        return cls(seqs, tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id)

    def allowed(self, prefix: Sequence[int]) -> list[int]:
        """Legal next tokens after `prefix`. Memoised: beams in a batch share most prefixes."""
        key = tuple(prefix)
        hit = self._memo.get(key)
        if hit is None:
            hit = self._memo[key] = self._allowed(key)
        return hit

    def _allowed(self, prefix: Sequence[int]) -> list[int]:
        node = self.root
        for t in prefix:
            if t == self.eos_id and _END in node:
                return [self.eos_id]  # finished title: only keep emitting EOS (padding for beams)
            node = node.get(int(t))
            if node is None:
                return []
        out = [t for t in node if t != _END]
        if _END in node:
            out.append(self.eos_id)
        return out

    def contains(self, seq: Sequence[int]) -> bool:
        node = self.root
        for t in seq:
            node = node.get(int(t))
            if node is None:
                return False
        return _END in node


def make_logits_processor(trie: TitleTrie, start: int):
    """HF LogitsProcessor: positions >= `start` (the padded prompt length) are
    constrained to the trie. Rows with no legal continuation are forced to EOS."""
    import torch
    from transformers import LogitsProcessor

    class TrieLogitsProcessor(LogitsProcessor):
        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
            gen = input_ids[:, start:].tolist()  # one GPU->CPU copy per step
            rows, cols = [], []
            for row, prefix in enumerate(gen):
                allowed = trie.allowed(prefix) or [trie.eos_id]
                rows.extend([row] * len(allowed))
                cols.extend(allowed)
            mask = torch.full_like(scores, float("-inf"))
            mask[torch.tensor(rows, device=scores.device), torch.tensor(cols, device=scores.device)] = 0.0
            return scores + mask  # a single scatter instead of one GPU op per beam

    return TrieLogitsProcessor()
