"""Item embeddings from an LLM: the "item understanding" half of LLM -> SASRec distillation.

Each catalog title is run through the LLM once; the mean of the last hidden layer
over the title's tokens is its vector. SASRec then uses these frozen vectors (through
a small trainable adapter) next to, or instead of, its ID embeddings. Items that are
rare or absent in the training interactions still get a meaningful representation.

    python scripts/rec_sasrec_distill.py embed --model checkpoints/rec_qwen3_0.6b_noreason/merged --tag sft06
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def title_embeddings(model_path: str, titles: Sequence[str], batch_size: int = 0) -> np.ndarray:
    """[len(titles), hidden] float32, mean-pooled over each title's tokens (padding excluded)."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    from distillops.memory import adaptive_chunks, max_sequences

    tok = AutoTokenizer.from_pretrained(model_path)
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    from distillops.precision import compute_dtype

    model = AutoModel.from_pretrained(model_path, dtype=compute_dtype(), device_map="cuda").eval()
    lengths = [len(tok(t, add_special_tokens=False)["input_ids"]) for t in titles]
    order = sorted(range(len(titles)), key=lambda i: -lengths[i])
    batch_size = batch_size or max_sequences(model, max(lengths) if lengths else 1)
    out = np.zeros((len(titles), model.config.hidden_size), dtype=np.float32)

    def chunk(a: int, b: int) -> None:
        idx = order[a:b]
        enc = tok([titles[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            h = model(**enc).last_hidden_state.float()  # [B, T, H]
        m = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1.0)
        out[idx] = pooled.cpu().numpy()

    adaptive_chunks(len(titles), batch_size, chunk)
    return out
