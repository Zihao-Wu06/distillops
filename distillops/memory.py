"""Hardware-agnostic batch sizing for generation.

Two layers, neither tuned to a particular GPU:

1. An initial estimate from the memory that is free *after* the model is loaded and
   the model's own config:
       KV cache per token = 2 (K and V) x layers x kv_heads x head_dim x bytes_per_value
   The cache grows token by token and is reallocated as it grows (old and new copies
   briefly coexist, and freed blocks fragment), so peak usage is ~2x the final cache.
2. `adaptive_chunks`: after every batch, if the peak memory went above `limit` of the
   card, halve the batch for the next ones; on an out-of-memory error, halve and retry.
   This matters because some drivers (e.g. Windows WDDM) do not raise out-of-memory but
   silently spill to system RAM, which makes generation 10-100x slower with no error.
"""

from __future__ import annotations

from typing import Any, Callable

GROWTH = 2  # peak / final KV-cache size, from cache reallocation while generating


# Config field names differ between architectures (Llama/Qwen: num_hidden_layers, GPT-2: n_layer,
# ChatGLM: num_layers, ...); the first one present wins.
_DIM_NAMES = {
    "layers": ("num_hidden_layers", "n_layer", "num_layers", "n_layers"),
    "heads": ("num_attention_heads", "n_head", "num_heads", "n_heads"),
    "kv_heads": ("num_key_value_heads", "n_kv_heads", "num_kv_heads"),
    "hidden": ("hidden_size", "n_embd", "d_model", "dim"),
    "head_dim": ("head_dim", "kv_channels"),
    "vocab": ("vocab_size", "padded_vocab_size"),
}


def model_dims(config: Any) -> dict[str, int]:
    """Layers, attention heads, KV heads, hidden size, head size and vocabulary of a causal LM config,
    whatever the architecture calls them. Raises ValueError if the essential ones are missing."""
    get_text = getattr(config, "get_text_config", None)  # multimodal configs nest the LM config
    cfg = get_text() if callable(get_text) else getattr(config, "text_config", config)

    def first(key: str) -> int | None:
        for name in _DIM_NAMES[key]:
            value = getattr(cfg, name, None)
            if isinstance(value, int) and value > 0:
                return value
        return None

    layers, heads, hidden = first("layers"), first("heads"), first("hidden")
    if not (layers and heads and hidden):
        raise ValueError(f"cannot read layers / heads / hidden size from {type(cfg).__name__}")
    if getattr(cfg, "multi_query", False) and not getattr(cfg, "new_decoder_architecture", False):
        kv_heads = 1  # multi-query attention (GPT-BigCode, Falcon-7B); Falcon-40B style lists its KV heads
    elif getattr(cfg, "multi_query_attention", False):
        kv_heads = getattr(cfg, "multi_query_group_num", 1)  # ChatGLM
    else:
        kv_heads = first("kv_heads") or heads
    return {"layers": layers, "heads": heads, "kv_heads": kv_heads, "hidden": hidden,
            "head_dim": first("head_dim") or hidden // heads, "vocab": first("vocab") or 50_000}


def kv_bytes_per_token(config: Any, bytes_per_value: int = 2) -> int:
    d = model_dims(config)
    return 2 * d["layers"] * d["kv_heads"] * d["head_dim"] * bytes_per_value


def bytes_per_sequence(config: Any, seq_len: int, beam: bool = False, bytes_per_value: int = 2) -> int:
    """Peak bytes for one generated sequence: KV cache (x GROWTH) plus per-step
    logits / mask buffers over the vocabulary. Beam search also copies the cache when it
    reorders beams, hence another x2 (measured: ~2x for greedy, ~3.7x for beams)."""
    per = kv_bytes_per_token(config, bytes_per_value) * seq_len * GROWTH + model_dims(config)["vocab"] * 16
    return per * (2 if beam else 1)


def max_sequences(
    model: Any,
    seq_len: int,
    beam: bool = False,
    fraction: float = 0.7,
    cap: int = 256,
    free_bytes: int | None = None,
) -> int:
    """How many sequences of `seq_len` tokens fit in `fraction` of free GPU memory.
    `free_bytes` can be passed explicitly (tests, or to share a GPU)."""
    if free_bytes is None:
        import torch

        if not torch.cuda.is_available():
            return 1
        free_bytes, _ = torch.cuda.mem_get_info()
        # Memory PyTorch has cached but is not using (e.g. after a previous phase) is
        # reusable too; the driver reports it as taken.
        free_bytes += torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
    itemsize = getattr(getattr(model, "dtype", None), "itemsize", 2)  # the KV cache uses the model's dtype
    try:
        per = bytes_per_sequence(model.config, seq_len, beam, itemsize)
    except ValueError as e:  # e.g. a model without attention heads: start small, adaptive_chunks adjusts
        print(f"note: {e}; starting generation with batch {min(cap, 8)}", flush=True)
        return min(cap, 8)
    return max(1, min(cap, int(free_bytes * fraction) // per))


def _is_oom(e: BaseException) -> bool:
    return "out of memory" in str(e).lower()


def adaptive_chunks(
    n: int,
    batch: int,
    run: Callable[[int, int], None],
    limit: float = 0.85,
    log: Callable[[str], None] = print,
    peak_fraction: Callable[[], float] | None = None,
) -> int:
    """Call run(start, end) over [0, n) in chunks, starting at `batch`.
    Halves the batch after a chunk whose peak memory exceeded `limit` of the card, and
    retries a chunk that raised out-of-memory with half the batch. Returns the final batch.
    `peak_fraction` (peak bytes / total bytes since the last call) is injectable for tests."""
    if peak_fraction is None:
        peak_fraction = _cuda_peak_fraction
    start = 0
    while start < n:
        end = min(n, start + batch)
        try:
            run(start, end)
        except RuntimeError as e:  # torch.cuda.OutOfMemoryError is a RuntimeError
            if not _is_oom(e) or batch == 1:
                raise
            batch = max(1, batch // 2)
            log(f"out of memory: retrying with batch {batch}")
            _empty_cache()
            continue
        start = end
        frac = peak_fraction()
        if frac > limit and batch > 1:
            batch = max(1, batch // 2)
            log(f"peak memory {frac:.0%} of the GPU (> {limit:.0%}): batch -> {batch}")
            _empty_cache()
    return batch


def _cuda_peak_fraction() -> float:
    import torch

    if not torch.cuda.is_available():
        return 0.0
    peak = torch.cuda.max_memory_reserved()
    torch.cuda.reset_peak_memory_stats()
    return peak / torch.cuda.get_device_properties(0).total_memory


def _empty_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def wait_for_free_memory(fraction: float, poll_s: float = 30.0, timeout_s: float = 6 * 3600,
                         log: Callable[[str], None] = print) -> None:
    """Block until `fraction` of the GPU's memory is free (e.g. another job sharing the card
    finishes). vLLM reserves a fixed fraction up front and fails if it is not available."""
    import time

    import torch

    if not torch.cuda.is_available():
        return
    start = time.time()
    while True:
        free, total = torch.cuda.mem_get_info()
        if free >= fraction * total or time.time() - start > timeout_s:
            return
        log(f"waiting for GPU memory: {free / total:.0%} free, need {fraction:.0%}")
        time.sleep(poll_s)
