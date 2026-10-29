# Recommendation example: an LLM teacher for a deployable sequential recommender

This example shows distillops on a task far from math: next-item recommendation on Amazon
Industrial & Scientific. A fine-tuned Qwen3-1.7B recommends well but is too large to serve, so it is
used offline as a teacher, and its knowledge is distilled into SASRec models of ~0.5M parameters.
Back to the [main README](../../README.md).

**Result:** two ~0.5M-parameter SASRecs reach **HR@10 .164 / NDCG@10 .119**, on par with recent LLM
recommenders (1.5B-7B parameters) at ~1/1,500 of their serving size; LLM item vectors plus soft labels
lift SASRec by 22-24% HR@10 at unchanged serving cost.

## Step 1: an LLM recommender as teacher (Amazon Industrial, 1,000 test users)
| Model | HR@10 | NDCG@10 |
|---|---|---|
| Popularity | .058 | .051 |
| SASRec (tuned) | .136 | .106 |
| Qwen3-0.6B, untrained | .103 | .061 |
| Qwen3-0.6B SFT on purchase histories | .138 | .097 |
| **Qwen3-1.7B SFT on purchase histories (teacher below)** | **.147** | **.1005** |

The LLM generates the next item's title under a catalog-trie constraint (no invalid items). It is
accurate but too large to serve, so it is used offline as the teacher for SASRec.

## Step 2: distilling the LLM into deployable SASRec (full test set, 4,533 users, 3 seeds)
```
Qwen3-1.7B (SFT on real purchases, HR@10 .147)
   ├─ item understanding: title embeddings ──┐
   └─ ranking knowledge: top-3 soft labels ──┴─► SASRec A' (recall top-20) ─┐
                                                 original SASRec ────────────┴─► 0.5/0.5 score blend -> top-10
```
| Model | Serving size | HR@3 | NDCG@3 | HR@5 | NDCG@5 | HR@10 | NDCG@10 |
|---|---|---|---|---|---|---|---|
| SASRec (our tuned baseline) | 0.5M | .102 | .093 | .115 | .098 | .134 | .104 |
| SASRec + LLM item vectors + ranking KD (A') | 0.5M | .107 | .090 | .129 | .099 | **.166** | .111 |
| **Two-stage: A' recall top-20 + SASRec re-rank** | **2 x 0.5M** | **.113** | **.101** | **.132** | **.109** | .164 | **.119** |
| *MiniOneRec (Oct 2025), as reported* | 1.5B-7B LLM | .114 | .101 | .132 | .108 | .159 | .117 |
| *S-GRec (Feb 2026), as reported* | 1.5B LLM | .116 | .103 | .136 | .111 | .163 | .120 |
| *SASRec, as reported in those papers* | 0.5M | .079 | .070 | .091 | .075 | .109 | .081 |

Hyper-parameters (teacher top-k, distillation weight, blend weight) are chosen on the validation set;
the test set is reported once. Same data splits as MiniOneRec; metrics follow its evaluation script,
but the papers generate semantic IDs while we rank titles, so the comparison is close, not exact.

Cold-start (HR@10 on test targets never seen in training, 147 users): original SASRec .000;
SASRec on LLM item vectors only .030; rarely seen items (1-5 occurrences, 354 users) .001 -> .032.
The two-stage system loses this (its re-ranker is ID-based); routing unseen items past the re-ranker is the
standard fix and is not yet measured.

## Reproduce
Run from the repository root, after the [install](../../README.md#install).
```bash
C=Industrial_and_Scientific
python scripts/rec_prepare.py --category $C          # data download, popularity baseline, first SASRec
python scripts/rec_sasrec_sweep.py --category $C     # tuned SASRec (16 configs, selected on validation)

# 1) the LLM teacher: Qwen3-1.7B SFT on 6,612 users' purchase histories
python scripts/rec_build_sft.py --variant noreason --match-row-ids tasks/rec/resources/industrial_sft_rows.txt
python -m distillops sft configs/sft_rec_qwen3.yaml   # Qwen3-1.7B, writes checkpoints/rec_qwen3_1.7b
python scripts/rec_eval_llm.py --model checkpoints/rec_qwen3_1.7b/merged --n 1000 --tag sft-1.7b

# 2) distil into SASRec: item vectors, then soft labels on users the teacher was not trained on
python scripts/rec_sasrec_distill.py embed --model Qwen/Qwen3-1.7B --tag base17
python scripts/rec_sasrec_distill.py topk  --model checkpoints/rec_qwen3_1.7b/merged --tag sft17 \
         --beams 10 --n 12000 --exclude data/sft/$C/noreason_matched.jsonl
python scripts/rec_sasrec_distill.py train --emb base17 --kd sft17 --kd-weight 1 --kd-topk 3 \
         --only fused-tied+kd --seeds 3 --name A_prime

# 3) two-stage recall + re-rank (blend weight chosen on validation), 3 seeds, then the summary table
for s in 42 43 44; do python scripts/rec_two_stage.py --seed $s --out results/$C/two_stage/seed$s.json; done
python scripts/rec_two_stage.py --summarise results/$C/two_stage
```
`rec_sasrec_distill.py train` also offers `--kd-power` (sharper rank weights) and variants `id`,
`id-tied`, `fused`, `fused-tied`, `llm-tied` (item representation) with or without `+kd`.

## Related work
DLLM2Rec (RecSys 2024) distils LLM-based
  recommenders into conventional ones with ranking and embedding distillation, the approach followed
  here; LLM-ESR and LLMEmb use LLM item representations to strengthen sequential recommenders; SLMRec
  (ICLR 2025) distils into a smaller LLM. Benchmark and baselines follow
  [MiniOneRec](https://github.com/AkaliKong/MiniOneRec); S-GRec (arXiv 2602.10606) is the most recent
  result we compare against.
