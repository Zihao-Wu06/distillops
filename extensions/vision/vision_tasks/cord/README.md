# Reading receipts (CORD-v2): a 2B student that ties its 32B teacher

The student reads a receipt photo into JSON: every item (name, count, price) and the subtotal, tax,
service, discount, total, cash and change. The teacher labels the training receipts with no human
label: its reading is accepted only when the receipt's own arithmetic adds up.

## Results
200 test receipts (CORD validation + test). Student: Qwen3-VL-2B-Instruct (LoRA + projector SFT,
then full GRPO). Teacher: Qwen3-VL-32B-Instruct-FP8, served by vLLM on the same GPU (no API cost).

| Model | Human labels used | Mean F1 | Receipts read perfectly | Total correct |
|---|---|---|---|---|
| 2B, untrained | 0 | 0.756 | 10.5% | 88.0% |
| 32B teacher, first prompt | 0 | 0.848 | 29.5% | 97.5% |
| 32B teacher, with naming notes | 0 | 0.896 | 41.0% | 96.5% |
| 2B, SFT on the teacher's labels (693 receipts) | 0 | 0.897 | 42.0% | 96.0% |
| **+ GRPO on a small labelled pool** | **300** | **0.905** | **49.5%** | 96.5% |
| 2B, SFT on gold labels (control) | 800 | 0.928 | 58.0% | 97.5% |

Paired over the same 200 receipts (bootstrap 95% CI, sign test):
- teacher-label SFT adds **+0.142** [+0.121, +0.164] over the untrained model (150 better, 19 worse),
  p = 2e-26, and ties its teacher: +0.001 [-0.011, +0.014], p = 1.0;
- GRPO adds +0.008 [-0.000, +0.016] mean F1 (37 better, 25 worse), p = 0.16, not significant; but
  perfectly read receipts go from 84 to 99 (20 gained, 5 lost), McNemar p = 0.004;
- gold labels for all 800 training receipts still add +0.023 [+0.010, +0.036] over GRPO, p = 0.002.
  Teacher-label SFT reaches 82% of the gold-label gain over the untrained model, 87% after GRPO.

F1 per field (SFT on teacher labels -> + GRPO; gold-label SFT):

| | name | count | price | subtotal | tax | service | discount | total | cash | change |
|---|---|---|---|---|---|---|---|---|---|---|
| teacher-label SFT | .771 | .886 | .947 | .945 | .956 | .885 | .667 | .977 | .919 | .962 |
| + GRPO | .781 | .916 | .935 | .926 | .966 | .939 | .710 | .982 | .962 | .958 |
| gold-label SFT | .804 | .946 | .955 | .970 | .939 | .898 | .741 | .985 | .969 | .950 |

GRPO mostly cut extra entries: precision 0.848 -> 0.876, recall 0.942 -> 0.928 (gold-label SFT:
0.899 and 0.942).

## What happened along the way
- **The label-free check works.** A reading is accepted when the items add up to the subtotal, the
  charges to the total, and cash minus total to the change. 719 of the 800 gold annotations pass it.
  Teacher readings that pass score 0.907 F1 (48% exact), those that fail 0.801 (3% exact); 693 of 800
  receipts got an accepted reading within 3 attempts.
- **Conventions mattered more than reading ability.** The teacher's first item names scored 0.53 F1:
  it read them correctly but kept quantities and item codes in the name and listed add-ons and section
  headings as items. Three notes in the teacher's prompt (`NOTES` in task.py, written by comparing its
  answers with training labels) raised its names to 0.81 and its mean F1 from 0.848 to 0.896 (p = 3e-6);
  the student trained on the new labels gained the same (+0.055, p = 1e-7). The student's prompt never
  changed.
- **GRPO** (reward = per-receipt F1 against human labels): 200 training receipts as the pool
  (`pools.py`), of which the pass-rate filter kept 139 (21.5% always perfect, 9% the same partial
  score every time). Checkpoints were chosen on 100 other training receipts the student never trained
  on (the teacher's rejected ones, so harder than average). At lr 1e-6 validation stayed flat for 24
  steps (stopped). At lr 3e-6 it peaked at step 10 (0.802 -> 0.813) and then fell to 0.79 while the
  training reward kept rising: 139 receipts over 4 epochs is enough to overfit, so validation-based
  selection mattered. 35 steps took 30 minutes.
- **Cost:** the teacher labelling took 1,026 calls (1.23M prompt and 191k completion tokens) on the
  rented GPU. SFT: 75 steps, about 12 minutes. The student reads a receipt in 0.54 s (vLLM, batched).

## Data
- **Source:** [naver-clova-ix/cord-v2](https://huggingface.co/datasets/naver-clova-ix/cord-v2):
  Indonesian shop and restaurant receipts. `train` 800 receipts (the teacher's input, the gold control,
  and the GRPO pools), `test` 200 (CORD validation + test), used only for evaluation.
- **Images** are sized to at most 1024x1024 pixels (`CordTask.max_pixels`) for the teacher, training,
  evaluation and GRPO alike: about 1,000 image tokens.
- **Score** (`score`, the GRPO reward): F1 over (field, value) pairs, items as a multiset of
  (name, count, price); names compared case- and whitespace-insensitively, amounts as numbers.
  `metrics.json` adds precision, recall, exact match, total accuracy and F1 per field.

## Run
From the repository root, on one GPU with about 80 GB (an RTX PRO 6000 here), after installing
distillops and this extension, plus vLLM. Tested with torch 2.8, transformers 4.57.1, TRL 0.24.0,
vLLM 0.11.0, Python 3.11.
```bash
export PYTHONPATH=extensions/vision VLLM_API_KEY=local
T=vision_tasks.cord.task:CordTask
S=Qwen/Qwen3-VL-2B-Instruct
python extensions/vision/vision_tasks/cord/prepare.py --src <snapshot of naver-clova-ix/cord-v2> --out data/cord
python -m vision_tasks.cord.check_rules --split train          # how many gold receipts pass the check
python -m distillops eval --task $T --split test --model $S --max-new-tokens 1024 --tag base

# the teacher, served locally (any OpenAI-compatible endpoint that takes images works)
vllm serve Qwen/Qwen3-VL-32B-Instruct-FP8 --served-model-name qwen3-vl-32b --max-model-len 8192 \
    --gpu-memory-utilization 0.90 --limit-mm-per-prompt '{"image": 1, "video": 0}' --api-key local --port 8000 &
python -m vision_tasks.cord.eval_teacher --teacher extensions/vision/configs/teacher_cord_qwen3vl_eval.yaml \
    --split test --task-arg notes=1 --out results/cord/eval/teacher-notes-test
python -m distillops_vision teacher --task $T --task-arg notes=1 \
    --teacher extensions/vision/configs/teacher_cord_qwen3vl.yaml --out results/cord/teacher_train_notes
# stop the teacher's server before training

python -m distillops_vision sft extensions/vision/configs/sft_cord_qwen3vl2b.yaml --model $S \
    --train_file results/cord/teacher_train_notes/accepted.jsonl --output_dir checkpoints/cord_teacher_notes
python -m distillops eval --task $T --split test --model checkpoints/cord_teacher_notes/merged \
    --max-new-tokens 1024 --tag sft_teacher_notes                                          # 0.897

# GRPO on 200 human-labelled receipts, checkpoints chosen on 100 others
python -m vision_tasks.cord.pools --accepted results/cord/teacher_train_notes/accepted.jsonl
G=extensions/vision/configs/grpo_cord_qwen3vl2b.yaml
python -m distillops_vision grpo $G --task $T --split rl --filter-samples 8 --filter-only
python -m distillops_vision grpo $G --task $T --split rl --keep-ids checkpoints/cord_grpo/pass_rate_kept.txt --epochs 4
python -m distillops eval --task $T --split test --model checkpoints/cord_grpo --max-new-tokens 1024 --tag grpo  # 0.905

# control: the same SFT on the gold labels of all 800 training receipts
python -m distillops labels --task $T --split train --out data/sft/cord/gold_train.jsonl
python -m distillops_vision sft extensions/vision/configs/sft_cord_qwen3vl2b.yaml --model $S \
    --train_file data/sft/cord/gold_train.jsonl --output_dir checkpoints/cord_gold
python -m distillops eval --task $T --split test --model checkpoints/cord_gold/merged --max-new-tokens 1024 --tag sft_gold

E=results/cord/eval
python -m vision_tasks.cord.compare $E/sft_teacher_notes/predictions.jsonl $E/grpo/predictions.jsonl
```
`check_rules --predictions <predictions.jsonl>` compares readings that pass the check with those that
fail it, `eval_teacher` scores the teacher's first answer on a split.

## Files
```
prepare.py      CORD parquet files -> data/cord/{train,test}.jsonl + images
receipt.py      parsing, amounts, F1, the arithmetic check (rule_check)
task.py         CordTask: prompts, teacher notes, verify, score, metrics
pools.py        the GRPO pool (rl) and validation (valid) splits, from the training receipts
eval_teacher.py the teacher's first answers on a split, scored
check_rules.py  the check on gold labels, and on the teacher's answers
compare.py      paired bootstrap and sign test between two predictions.jsonl
../../configs/*cord*   teacher, SFT and GRPO settings
```
