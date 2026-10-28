"""Two-phase generation for an LLM recommender (HF transformers).

Phase 1 (reasoning, optional): greedy free-text generation until </think>.
Phase 2 (recommendation): beam search constrained to the title trie; the K
returned beams are K real catalog items, ranked by sequence log-probability.

`recommend_all` runs each phase with its own batch size (the reasoning phase can
batch ~20x more prompts than the beam phase).
"""

from __future__ import annotations

from typing import Sequence

from .prompts import THINK_CLOSE, student_messages
from .trie import TitleTrie, make_logits_processor


def im_end_id(tokenizer) -> int:
    tid = tokenizer.convert_tokens_to_ids("<|im_end|>")
    return tid if isinstance(tid, int) and tid >= 0 else tokenizer.eos_token_id


def build_trie(titles: Sequence[str], tokenizer) -> TitleTrie:
    seqs = [tokenizer(t, add_special_tokens=False)["input_ids"] for t in dict.fromkeys(titles)]
    return TitleTrie(seqs, im_end_id(tokenizer))


def check_boundary(tokenizer, titles: Sequence[str]) -> None:
    """The trie stores titles tokenized on their own; at generation time they follow
    '</think>\\n\\n'. Assert BPE does not merge across that boundary."""
    prefix = f"{THINK_CLOSE}\n\n"
    p = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    for t in list(titles)[:200]:
        joint = tokenizer(prefix + t, add_special_tokens=False)["input_ids"]
        alone = tokenizer(t, add_special_tokens=False)["input_ids"]
        assert joint == p + alone, f"token boundary merge for title: {t!r}"


def prompt_texts(tokenizer, histories: Sequence[Sequence[str]], category_name: str, think: bool) -> list[str]:
    return [
        tokenizer.apply_chat_template(
            student_messages(h, category_name), tokenize=False, add_generation_prompt=True, enable_thinking=think
        )
        for h in histories
    ]


def _left_pad(tokenizer) -> None:
    tokenizer.padding_side = "left"  # decoder-only batching: generation starts aligned
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


def generate_reasons(model, tokenizer, texts: Sequence[str], max_reason_tokens: int = 256) -> tuple[list[str], list[str]]:
    """Phase 1: greedy free reasoning until </think>. Returns (prompts extended with the
    closed reasoning block, reasoning text). One sequence per prompt, so this phase can
    use a much larger batch than the beam phase."""
    import torch

    _left_pad(tokenizer)
    enc = tokenizer(list(texts), return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **enc, max_new_tokens=max_reason_tokens, do_sample=False, temperature=None, top_p=None, top_k=None,
            stop_strings=[THINK_CLOSE], tokenizer=tokenizer, pad_token_id=tokenizer.pad_token_id,
        )
    gen = tokenizer.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=False)
    extended, reasons = [], []
    for t, g in zip(texts, gen):
        body = g.split(THINK_CLOSE, 1)[0].replace(tokenizer.pad_token, "")
        reasons.append(body.replace("<think>", "").strip())
        extended.append(t + body + f"{THINK_CLOSE}\n\n")  # truncated reasoning is closed for it
    return extended, reasons


def rank_titles(
    model, tokenizer, trie: TitleTrie, texts: Sequence[str], num_beams: int = 20, max_title_tokens: int | None = None
) -> list[list[str]]:
    """Phase 2: beam search constrained to the title trie; returns num_beams catalog titles per prompt."""
    import torch

    _left_pad(tokenizer)
    enc = tokenizer(list(texts), return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    start = enc["input_ids"].shape[1]
    proc = make_logits_processor(trie, start)
    # A fixed cap (e.g. 64) silently truncates long titles into invalid items.
    max_title_tokens = max_title_tokens or trie.max_depth + 1
    with torch.no_grad():
        out = model.generate(
            **enc,
            num_beams=num_beams,
            num_return_sequences=num_beams,
            max_new_tokens=max_title_tokens,
            do_sample=False, temperature=None, top_p=None, top_k=None,
            length_penalty=0.0,  # rank by total log-prob, no length normalisation (as MiniOneRec)
            early_stopping=True,
            logits_processor=[proc],
            eos_token_id=trie.eos_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    titles = [t.strip() for t in tokenizer.batch_decode(out[:, start:], skip_special_tokens=True)]
    return [titles[i * num_beams : (i + 1) * num_beams] for i in range(len(texts))]


def recommend(
    model,
    tokenizer,
    trie: TitleTrie,
    histories: Sequence[Sequence[str]],
    category_name: str,
    think: bool,
    num_beams: int = 20,
    max_reason_tokens: int = 256,
    max_title_tokens: int | None = None,
) -> tuple[list[list[str]], list[str]]:
    """One batch, both phases. Returns (ranked titles per history, reasoning text per history)."""
    texts = prompt_texts(tokenizer, histories, category_name, think)
    reasons = [""] * len(texts)
    if think:
        texts, reasons = generate_reasons(model, tokenizer, texts, max_reason_tokens)
    return rank_titles(model, tokenizer, trie, texts, num_beams, max_title_tokens), reasons


def recommend_all(
    model,
    tokenizer,
    trie: TitleTrie,
    histories: Sequence[Sequence[str]],
    category_name: str,
    think: bool,
    num_beams: int = 20,
    beam_batch: int = 0,
    reason_batch: int = 0,
    max_reason_tokens: int = 256,
    max_title_tokens: int | None = None,
) -> tuple[list[list[str]], list[str]]:
    """All histories, each phase with its own batch size (order preserved).

    Batch sizes of 0 start from an estimate based on free GPU memory and the model config,
    and shrink automatically if a batch comes close to the card's limit
    (distillops.memory), so the same code adapts to any GPU. The reasoning phase has one
    sequence per prompt and the beam phase `num_beams`, so it batches far more prompts."""
    from distillops.memory import adaptive_chunks, max_sequences

    texts = prompt_texts(tokenizer, histories, category_name, think)
    prompt_len = max(len(tokenizer(t, add_special_tokens=False)["input_ids"]) for t in texts) if texts else 0
    title_len = max_title_tokens or trie.max_depth + 1
    reason_len = max_reason_tokens if think else 0
    reasons = [""] * len(texts)

    if think:
        reason_batch = reason_batch or max_sequences(model, prompt_len + reason_len)
        print(f"reasoning: start batch {reason_batch} prompts", flush=True)
        order = sorted(range(len(texts)), key=lambda i: -len(texts[i]))  # similar lengths share a batch

        def reason_chunk(a: int, b: int) -> None:
            idx = order[a:b]
            ext, rs = generate_reasons(model, tokenizer, [texts[i] for i in idx], max_reason_tokens)
            for i, t, r in zip(idx, ext, rs):
                texts[i], reasons[i] = t, r

        adaptive_chunks(len(texts), reason_batch, reason_chunk)
        print(f"reasoning done for {len(texts)} prompts", flush=True)

    beam_batch = beam_batch or max(1, max_sequences(model, prompt_len + reason_len + title_len, beam=True) // num_beams)
    print(f"beam search: start batch {beam_batch} prompts x {num_beams} beams", flush=True)
    ranked: list[list[str] | None] = [None] * len(texts)

    def beam_chunk(a: int, b: int) -> None:
        ranked[a:b] = rank_titles(model, tokenizer, trie, texts[a:b], num_beams, max_title_tokens)
        if b // 200 != a // 200:
            print(f"ranked {b}/{len(texts)}", flush=True)

    adaptive_chunks(len(texts), beam_batch, beam_chunk)
    return ranked, reasons  # type: ignore[return-value]
