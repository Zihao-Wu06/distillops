"""Hardware-agnostic batch sizing, and the two-phase recommender keeping order."""

from types import SimpleNamespace

import tasks.rec.generate as gen
from distillops.memory import adaptive_chunks, kv_bytes_per_token, max_sequences

QWEN3_06B = SimpleNamespace(num_hidden_layers=28, num_attention_heads=16, num_key_value_heads=8,
                            head_dim=128, hidden_size=1024, vocab_size=151936)


def test_kv_cache_size_from_config():
    assert kv_bytes_per_token(QWEN3_06B) == 2 * 28 * 8 * 128 * 2  # ~112 KiB per token in bf16


def test_max_sequences_scales_with_free_memory_and_beams():
    model = SimpleNamespace(config=QWEN3_06B)
    small = max_sequences(model, 600, free_bytes=8 * 2**30)
    big = max_sequences(model, 600, free_bytes=80 * 2**30, cap=10_000)
    assert 1 <= small < big
    assert max_sequences(model, 600, beam=True, free_bytes=8 * 2**30) < small  # beam search reorders the cache
    assert max_sequences(model, 600, free_bytes=1) == 1  # never zero


def test_recommend_all_preserves_order_across_phase_batches(monkeypatch):
    histories = [["a"] * k for k in (1, 5, 2, 4, 3)]  # different prompt lengths -> reordered in phase 1
    monkeypatch.setattr(gen, "prompt_texts", lambda tok, hs, name, think: [" ".join(h) for h in hs])
    monkeypatch.setattr(gen, "generate_reasons",
                        lambda m, tok, texts, n: ([t + "|r" for t in texts], [f"reason-{len(t.split())}" for t in texts]))
    monkeypatch.setattr(gen, "rank_titles",
                        lambda m, tok, trie, texts, nb, mt: [[f"title-{t}"] for t in texts])
    monkeypatch.setattr("distillops.memory._cuda_peak_fraction", lambda: 0.1)
    tok = lambda t, add_special_tokens=False: {"input_ids": t.split()}  # noqa: E731
    trie = SimpleNamespace(max_depth=3)
    ranked, reasons = gen.recommend_all(None, tok, trie, histories, "x", think=True, num_beams=2,
                                        beam_batch=2, reason_batch=2)
    assert reasons == [f"reason-{len(h)}" for h in histories]
    assert ranked == [[f"title-{' '.join(h)}|r"] for h in histories]


def test_adaptive_chunks_halves_after_a_batch_near_the_memory_limit():
    calls = []
    peaks = iter([0.95, 0.5, 0.5, 0.5, 0.5, 0.5])  # first batch came too close to the limit
    final = adaptive_chunks(20, 8, lambda a, b: calls.append((a, b)), log=lambda m: None,
                            peak_fraction=lambda: next(peaks))
    assert calls[0] == (0, 8) and calls[1] == (8, 12) and final == 4
    assert sorted(i for a, b in calls for i in range(a, b)) == list(range(20))  # every item exactly once


def test_adaptive_chunks_retries_out_of_memory_with_half_the_batch():
    calls = []

    def run(a, b):
        calls.append((a, b))
        if b - a > 2:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")

    adaptive_chunks(5, 8, run, log=lambda m: None, peak_fraction=lambda: 0.1)
    assert calls[:3] == [(0, 5), (0, 4), (0, 2)]
    assert [c for c in calls if c[1] - c[0] <= 2] == [(0, 2), (2, 4), (4, 5)]
