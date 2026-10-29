# PII example: a small personal-information detector distilled from an API LLM

Find personal information (names, contact details, addresses, ID and card numbers, dates,
demographics) in English text so it can be masked before the text is stored, shared or sent to
another model. A strong API model labels the text; a 0.6B open model learns to do the same job
locally, without the data leaving the machine. Back to the [main README](../../README.md).

**Result:** the 0.6B student **never trains on a human label**, yet matches and then beats its
teacher. The only labels behind it are the teacher's: the gold labels of 100 training texts, used to
write its instructions, and 3 worked examples. On 2,000 held-out texts it reaches **0.955 F1**
(teacher 0.934), and the share of personal information left visible after masking falls from 73% to
**1.9%** (teacher 3.0%). Teacher calls cost 17.7 CNY.

The task is one `Task` class (`tasks/pii/task.py`) plus data preparation; teacher calls, rejection
sampling, SFT, evaluation and GRPO are the framework's.

## Results
English test set, 2,000 texts, 7 categories. Student: Qwen3-0.6B (LoRA SFT, then full GRPO).
Teacher: DeepSeek `deepseek-flash` with 3 worked examples.

| Model | Labelled texts used | Mean F1 | Micro-F1 | Leak rate |
|---|---|---|---|---|
| Qwen3-0.6B, untrained | - | 0.158 | 0.224 | 73.1% |
| Teacher (DeepSeek flash, 3-shot) | 103 (100 to write its instructions, 3 as examples) | 0.934 | 0.938 | 3.0% |
| **0.6B, SFT on rule-checked teacher labels** | **103** (the teacher's; it trains on none) | **0.945** | 0.948 | 2.0% |
| **+ GRPO on a small labelled pool** | **2,100** (+ the 2,000-text pool, which includes the 3 examples) | **0.955** | 0.955 | **1.9%** |
| 0.6B, SFT on gold labels (control) | 10,000 | 0.982 | 0.983 | 1.1% |

Paired over the same 2,000 texts (bootstrap 95% CI, sign test):
- the student trained on teacher labels beats its teacher: **+1.15 points** [+0.65, +1.67], p = 4e-5;
- GRPO adds **+0.95** [+0.64, +1.27] over its SFT start (better on 127 texts, worse on 47), p = 1e-9;
- a complete human annotation of the same 10,000 texts would still add 2.75 points [+2.34, +3.18]:
  the teacher's labels reach 96% of gold-label quality after SFT, 97% after GRPO.

F1 per category:

| Model | NAME | CONTACT | ADDRESS | ID_NUMBER | CARD_NUMBER | DATE | DEMOGRAPHIC |
|---|---|---|---|---|---|---|---|
| Teacher | .939 | .973 | .854 | .959 | .929 | .992 | .949 |
| SFT on teacher labels | .963 | .972 | .864 | .973 | .946 | .994 | .955 |
| + GRPO | .964 | .975 | .891 | .975 | .953 | .992 | .958 |
| SFT on gold labels | .985 | .986 | .978 | .986 | .971 | .994 | .969 |

What happened along the way:
- **Teacher labelling:** 9,681 of 10,000 texts passed the rules (96.8%; 9,518 at the first try),
  10,844 calls in 70 seconds at 256 parallel requests. Rejections: a 9+ digit number left
  unmasked (588), a value not copied verbatim (421), a "card number" with too few digits (133).
- **Guidelines from 100 checked outputs.** Before labelling at scale, the teacher's answers on
  100 training texts were compared with their labels, as an annotation team would with a pilot
  batch. Three rounds of instructions (adjacent words of one category form one entry; any city
  counts; an age is just the number) raised the teacher's F1 on that batch from 0.878 to 0.935
  and cut its leak rate from 8.9% to 1.9%.
- **GRPO** (reward = per-text F1): the pass-rate filter kept the 850 of 1,997 labelled prompts whose
  8 samples differ in score; 49% were always right and **7.9% got the same partial score every
  time** (no learning signal, so excluded). 54 steps (2 epochs); validation F1 0.952 -> 0.965,
  best at the last step; weight change 2.6e-4 (floor 2.6e-5); no entropy collapse.
- **Leakage checks:** no test text appears in another split; 3 of the 2,000 closely resemble a
  training text (>= 50% of 8-word sequences in common). GRPO validation used its own 300-text split,
  and the teacher answered the test set once, with its final instructions.
- **Cost:** 13,054 teacher calls in total (pilot batches, labelling, the teacher on the test set),
  84% of input tokens served from DeepSeek's prompt cache: 8.85 CNY at off-peak prices, **17.7 CNY**
  as run (peak hours, 2x).
- **Speed:** the student labels a text in 0.12 s (batched greedy decoding with plain transformers on
  one GPU; a serving engine would be faster), with nothing sent outside the machine.

These results were produced with distillops `12eb80e`, before the task moved into this repository.
The port was checked against that run: re-scoring the saved outputs of all five models above gives
identical metrics, and the teacher prompts are byte-identical (they hit the original run's response
cache). The pipeline has not been re-run end to end with the current version.

## Setting
- **Data:** [OpenPII-1M](https://huggingface.co/datasets/ai4privacy/pii-masking-openpii-1m)
  (ai4privacy, CC-BY-4.0, synthetic PII), English rows only (143,515 train, 35,904 validation).
- **Categories:** the dataset's 19 labels merged into 7 (`NAME`, `CONTACT`, `ADDRESS`, `ID_NUMBER`,
  `CARD_NUMBER`, `DATE`, `DEMOGRAPHIC`). Masking decides what is hidden, so the fine type (given
  name vs surname, ID card vs tax number) rarely matters, and it is where annotators disagree most.
  With 7 categories, adjacent spans of one category separated only by spaces are one entity
  ("Mr John Smith", "12 Oak Street", "Boston 02108"): where a given name ends and a surname starts
  is often unknowable from the text. `--task-arg labels=fine` keeps the 19 labels and their spans.
- **Splits** (disjoint, `scripts/pii_prepare.py`): `train` 10,000 texts for the teacher, `rl` 2,000
  labelled texts for GRPO (the teacher's 3 worked examples come from here and are excluded),
  `valid` 300 for GRPO checkpoint selection, `test` 2,000 from the dataset's validation split.
- **Output:** a JSON list `[{"type": ..., "value": ...}]` of values copied from the text; masking
  replaces every occurrence of each value.
- **No labels for the teacher's data.** As with a company's own text, a teacher answer is accepted
  by rules alone (`verify=rules`): known categories, every value copied verbatim, plausible formats
  (an email has an @ and a dotted domain, a card number 12+ digits, a phone number 7+, a date a
  digit or month), and no 9+ digit number left unmasked. On the training texts' gold labels these
  rules wrongly reject 0.17% of texts (numbers the annotation itself left out). Besides that check,
  gold labels are used only for the 100-text pilot batch (above), the teacher's 3 worked examples,
  the fully-labelled control, the GRPO pool and evaluation. `--task-arg verify=gold` requires an
  exact match instead.
- **Metrics** (`results/pii/eval/<tag>/metrics.json`): `score` = mean per-text F1 over
  (category, value) pairs, also the GRPO reward; corpus precision / recall / micro- and macro-F1;
  F1 per category; type-agnostic value F1; **leak rate** = share of the dataset's gold PII spans
  that stay (partly) visible after masking with the predicted values.

## Reproduce
Run from the repository root, after the [install](../../README.md#install) (GRPO needs Linux and
vLLM). Download OpenPII-1M's `train.jsonl` and `validation.jsonl` first.
```bash
T=tasks.pii.task:PIITask
TEACHER=configs/teacher_deepseek_pii.yaml     # any OpenAI-compatible teacher; export its key
S=Qwen/Qwen3-0.6B

python scripts/pii_prepare.py --src /path/to/pii-masking-openpii-1m/data      # data/pii/{train,rl,valid,test}.jsonl
python -m distillops eval    --task $T --split test --model $S --no-think --tag base                # 0.158
python -m distillops teacher --task $T --teacher $TEACHER --n 100                                  # smoke test
python -m distillops teacher --task $T --teacher $TEACHER                                          # 10,000 texts, rules only
python scripts/pii_eval_teacher.py --teacher $TEACHER --split test                                  # the teacher: 0.934
python -m distillops sft     configs/sft_pii_qwen3.yaml
python -m distillops eval    --task $T --split test --model checkpoints/pii_0.6b_teacher/merged --no-think --tag sft-teacher  # 0.945

# GRPO on the 2,000-text labelled pool, reward = per-text F1
python -m distillops grpo configs/grpo_pii_qwen3.yaml --task $T --split rl --filter-samples 8 --filter-only
python -m distillops grpo configs/grpo_pii_qwen3.yaml --task $T --split rl \
         --keep-ids checkpoints/pii_0.6b_grpo/pass_rate_kept.txt --epochs 2
python -m distillops eval    --task $T --split test --model checkpoints/pii_0.6b_grpo --no-think --tag grpo  # 0.955

# control: the same SFT on the gold labels of the 10,000 texts
python -m distillops labels  --task $T
python -m distillops sft     configs/sft_pii_qwen3.yaml --train_file data/sft/pii/labels_train_n0.jsonl \
         --output_dir checkpoints/pii_0.6b_gold
python -m distillops eval    --task $T --split test --model checkpoints/pii_0.6b_gold/merged --no-think --tag sft-gold  # 0.982
python -m distillops usage   --teacher $TEACHER
```
`python scripts/pii_check_rules.py --split train` shows how often the rules reject gold answers.
Tested on one RTX PRO 6000 (96 GB) with torch 2.8.0, transformers 4.57.1, TRL 0.24.0, vLLM 0.11.0.

## Limitations
- The data is synthetic and mostly formal text (letters, forms, notices, chats); scores are
  agreement with this dataset's annotation conventions, not with every organisation's policy.
- The rules cannot see a missed name or address; the teacher's misses become the student's. The
  pilot-batch check and GRPO on a small labelled pool are what close that gap here.

## Files
```
tasks/pii/labels.py         the 19 labels and the 7 categories
tasks/pii/entities.py       parsing replies, F1, masking, leak counting, label-free rule checks
tasks/pii/task.py           PIITask: prompts, verification, training rows, score, corpus metrics
scripts/pii_prepare.py      English splits from OpenPII-1M
scripts/pii_check_rules.py  how often the rules reject the gold answer
scripts/pii_eval_teacher.py the teacher on a split, same metrics as the student
configs/*pii*               teacher, SFT and GRPO settings
tests/test_pii_*.py         unit tests (no network or GPU)
```

Data: OpenPII-1M by Ai4Privacy (CC-BY-4.0).
