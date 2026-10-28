# distillops — distil an API LLM into a small model, for any task with a checkable answer

distillops turns a strong API model (the *teacher*) into a small open model you can run yourself,
for **any task where a program can tell a good answer from a bad one**. You describe the task once,
in a `Task` class; the same commands then collect verified teacher data, fine-tune, evaluate and run
reinforcement learning. The framework never needs the teacher's logits, only its text answers.

## What it can be used for
The one requirement: given an input and an output, your code can say whether the output is right
(`verify`, for teacher data) and how right it is on a 0-1 scale (`score`, for evaluation and RL).

| Task type | What `verify` / `score` checks |
|---|---|
| Classification, routing, intent detection | the predicted label equals the gold label |
| Structured extraction (JSON, forms, entities) | the output parses and its fields match |
| Tool / function calling | the function name and arguments match the expected call |
| Code generation | the code passes the task's unit tests |
| Math and quantitative reasoning | the final number equals the answer |
| Short-answer QA | the normalised answer matches |
| Ranking, retrieval, recommendation | the right item appears in the output (e.g. in the top-k) |

Two of these ship as worked examples, chosen to be very different: **math (GSM8K)** and
**recommendation (Amazon)**. They show what the framework does and how well; nothing in the framework
is specific to them. Any other task needs only its own `Task` class
([Add your own task](#add-your-own-task) has a complete one in ~45 lines).

## How it works
```
         ┌───────────────────────────────────────────────────────────────┐
 task ──►│ Task class: data, teacher prompt, verify, training row,       │  the only task-specific code
         │ student prompt, score (+ optional: label-only rows, decoding) │
         └───────────────────────────────────────────────────────────────┘
                                    │
 0 check     python -m distillops doctor     GPU, chosen precision, free memory and disk, versions
                                    │
 1 baseline  python -m distillops eval       the untrained student: the starting point
                                    │
 2 teacher   python -m distillops teacher    any OpenAI-compatible API answers -> verify -> retry the failed
                                             ones (up to --max-attempts, default 3): rejection sampling
   control   python -m distillops labels     rows from the dataset's own labels (answers or human
                                             solutions), no teacher
                                    │
 3 SFT       python -m distillops sft        one run on teacher rows, one on control rows (LoRA by default)
                                    │
 4 eval      python -m distillops eval       teacher SFT vs control SFT vs baseline
                                    │
 5 GRPO      python -m distillops grpo       reward = Task.score; optional --filter-samples keeps prompts that
                                             are sometimes right; validation picks the best checkpoint;
                                             stops on entropy collapse
                                    │
 6 eval      python -m distillops eval       before vs after GRPO
```

`verify` decides which teacher outputs become training data; `score` is both the evaluation metric
and the GRPO reward, so reinforcement learning optimises exactly what you report.

## Highlights
- **Any verifiable task, one class.** Teacher calls, rejection sampling, SFT, evaluation and GRPO are
  shared; a task only supplies data, prompts, a verifier and a score.
- **Verified data, measured value.** Teacher outputs are kept only if they pass your verifier (failed
  examples are retried), and a no-teacher control (`labels`) shows what the teacher actually adds.
- **RL that tells you when it will not work.** Before GRPO, the share of prompts that carry a learning
  signal is measured and explained; during and after it, validation picks the best checkpoint, an
  entropy guard stops collapse, and a weight-change check catches updates that never landed.
- **Runs on the GPU you have.** Precision is chosen from the hardware, batch sizes from free memory and
  the model's size, and disk space is checked before training; `doctor` reports all of it.
- **Example results.** GSM8K: a 0.6B student goes from 62.4% to **66.8%** (teacher reasoning beats
  human-written solutions by 13.5 points; 3.76 CNY of teacher calls). Recommendation: a fine-tuned 1.7B
  model distilled into two ~0.5M-parameter SASRecs reaches **HR@10 .164 / NDCG@10 .119**, on par with
  recent LLM recommenders at ~1/1,500 of their serving size.

## Contents
[Install](#install) · [Quick start](#quick-start) · [Add your own task](#add-your-own-task) ·
[Commands](#commands) · [Configuration](#configuration) · [Outputs](#outputs) ·
[What the framework handles](#what-the-framework-handles) · [Safeguards](#safeguards) ·
[Included examples and results](#included-examples-and-results) · [Related work](#related-work) · [Layout](#layout)

## Install
```bash
pip install -e ".[train,dev]"          # framework, training stack and tests
pip install "vllm==0.11.0"             # only for GRPO and the pass-rate filter (Linux)
python -m distillops doctor            # GPU, chosen precision, free memory and disk, library versions
python -m pytest -q                    # unit tests; no GPU or network needed
```
Python 3.11-3.12. Tested on two setups:

| Machine | Used for | torch | transformers | TRL | PEFT | vLLM |
|---|---|---|---|---|---|---|
| Windows 11, RTX 4090 24GB | teacher calls, SFT, evaluation, SASRec | 2.11 (cu128) | 5.21 | 1.16 | 0.21 | - |
| Ubuntu 22.04, RTX 6000D 84GB | GRPO, pass-rate filter, SFT, evaluation | 2.8 (cu128) | 4.57.1 | 0.24 | 0.21 | 0.11.0 |

vLLM 0.11 needs transformers 4.x; pin `transformers==4.57.1 trl==0.24.0` next to it. The framework
handles the version differences it knows about (warmup arguments, chat-template options).

**Teacher terms of use:** check that your teacher's terms allow training other models on its outputs.
DeepSeek's do; some providers' terms forbid training competing models.

## Quick start
The whole workflow on the included GSM8K example. For your own task, set `T` to its import path; the
commands stay the same.
```bash
export DEEPSEEK_API_KEY=...
T=tasks.math.gsm8k:GSM8KTask

python -m distillops eval    --task $T --model Qwen/Qwen3-0.6B --no-think           # baseline: 62.4%
python -m distillops teacher --task $T --n 100                                     # smoke test (~0.05 CNY)
python -m distillops teacher --task $T                                             # full train split (3.76 CNY)
python -m distillops labels  --task $T                                             # control: answers only
python -m distillops sft     configs/sft_qwen3.yaml \
         --train_file results/gsm8k/teacher_train_n0/accepted.jsonl --output_dir checkpoints/gsm8k_0.6b_teacher
python -m distillops eval    --task $T --model checkpoints/gsm8k_0.6b_teacher/merged # 62.5%

# GRPO (Linux + vLLM): measure the learning signal first, then train on the useful prompts
python -m distillops grpo configs/grpo_gsm8k_qwen3_0.6b.yaml --task $T --filter-samples 8 --filter-only
python -m distillops grpo configs/grpo_gsm8k_qwen3_0.6b.yaml --task $T \
         --keep-ids checkpoints/gsm8k_0.6b_grpo/pass_rate_kept.txt --epochs 1
python -m distillops eval    --task $T --model checkpoints/gsm8k_0.6b_grpo           # 66.8%
python -m distillops cost                                                          # teacher spend so far
```

## Add your own task
A complete task: support tickets classified into a fixed label set, with the teacher explaining its
choice. Save it as e.g. `my_tasks/triage.py`.

```python
import json
import re

from distillops.task import Task, Verdict


class TicketTriage(Task):
    name = "triage"
    LABELS = ["billing", "bug", "feature", "other"]

    def load_examples(self, split):  # any source; here JSONL rows {"id", "text", "label"}
        with open(f"data/triage_{split}.jsonl", encoding="utf-8") as f:
            return [json.loads(line) for line in f]

    def example_id(self, ex):
        return ex["id"]

    def teacher_messages(self, ex, attempt):
        return [{"role": "user", "content": f"{ex['text']}\n\nExplain in one or two sentences which "
                 f"team should handle this, then end with: LABEL: <one of {', '.join(self.LABELS)}>"}]

    def parse_teacher(self, ex, text):
        found = re.findall(r"LABEL:\s*(\w+)", text)
        return found[-1].lower() if found else None

    def verify(self, ex, parsed, text, attempt):
        if parsed not in self.LABELS:
            return Verdict(False, "parse_error")
        return Verdict(parsed == ex["label"], "ok" if parsed == ex["label"] else "wrong_label")

    def sft_example(self, ex, parsed, text):
        return {"prompt": self.student_messages(ex), "completion": [{"role": "assistant", "content": text}]}

    def student_messages(self, ex):
        return [{"role": "user", "content": f"{ex['text']}\n\nEnd with: LABEL: <one of {', '.join(self.LABELS)}>"}]

    def score(self, ex, completion):
        return float(self.parse_teacher(ex, completion) == ex["label"])

    def gold_sft_example(self, ex):  # optional: the no-teacher control for `labels`
        return {"prompt": self.student_messages(ex),
                "completion": [{"role": "assistant", "content": f"LABEL: {ex['label']}"}]}
```

```bash
python scripts/check_task.py --task my_tasks.triage:TicketTriage      # data, prompts, masked SFT row: check before paying
python -m distillops teacher --task my_tasks.triage:TicketTriage --n 50 # then labels / sft / eval / grpo as in Quick start
```

The methods, with the included GSM8K example for comparison:

| Method | Purpose | GSM8K |
|---|---|---|
| `load_examples(split)` | your dataset (a `valid` split, if any, is used for GRPO validation) | `datasets.load_dataset("openai/gsm8k")` |
| `example_id(ex)` | stable id (logging, resume, `--exclude`, `--keep-ids`) | `"gsm8k:train:17"` |
| `teacher_messages(ex, attempt)` | prompt for the teacher | question + "end with `ANSWER: <number>`" |
| `parse_teacher(ex, text)` | raw text -> structured answer | steps + final number |
| `verify(ex, parsed, text, attempt)` | **the distillation filter** | number equals gold, length limit |
| `sft_example(ex, parsed, text)` | accepted output -> training row | `<think>steps</think>ANSWER: n` |
| `student_messages(ex)` | the student's prompt | question + answer-format instruction |
| `score(ex, completion)` | metric **and** GRPO reward, in [0, 1] | 1 if the final number is right |
| `gold_sft_example(ex)` *(optional)* | label-only control row | `ANSWER: n`, or the human solution |
| `evaluate(...)` *(optional)* | custom decoding | - (recommendation overrides it with constrained beam search) |

Constructor arguments can be passed on the command line, e.g. `--task-arg max_words=200`.

**Writing a good verifier is most of the work.** Teachers break instructions in ways that would
teach the student the wrong thing (e.g. referring to information only the teacher saw); `verify` is
where those outputs are dropped. Read accepted samples before scaling up. If GRPO's pass-rate filter
reports a sparse reward, a `score` that gives partial credit is the first thing to try.

## Commands
`python -m distillops <command> --help` lists every flag. Commands that read a dataset share
`--task module:Class` (or a YAML file with `task:` and `args:`), `--task-arg KEY=VALUE` (constructor
arguments, repeatable), `--split`, `--n` (random subset, 0 = all) and `--seed`.

| Command | What it does | Main flags |
|---|---|---|
| `doctor` | environment report and platform warnings | - |
| `teacher` | teacher outputs -> `verify` -> accepted SFT rows, with retries | `--teacher` (config), `--max-attempts` (3), `--out`, `--allow-peak` |
| `labels` | label-only SFT rows from `gold_sft_example` (the no-teacher control) | `--out` |
| `sft` | supervised fine-tuning from a YAML config | `--train_file`, `--output_dir`, `--model`, `--epochs` (override the config) |
| `eval` | generate with a student and score with `Task.score` | `--model`, `--max-new-tokens` (512), `--batch` (0 = auto), `--no-think`, `--tag`, `--out` |
| `grpo` | GRPO with `Task.score` as the reward | `--filter-samples N`, `--filter-only`, `--filter-gpu-memory` (0.4), `--keep-ids`, `--exclude`, `--epochs`, `--max-steps`, `--eval-every`, `--eval-examples`, `--output_dir` |
| `merge` | merge a LoRA adapter into its base model | `--base`, `--adapter`, `--out` |
| `cost` | teacher spend from the local response cache | `--teacher` |

`grpo` details: `--filter-samples 8 --filter-only` samples 8 completions per prompt with vLLM, prints
how many prompts give a learning signal, and writes `pass_rate_kept.txt`; `--keep-ids` then trains on
those prompts in a fresh process. `--exclude <jsonl>` skips prompts whose `meta.id` appears in a file
(e.g. the SFT data). `--epochs E` sets the number of steps so each prompt is seen E times.

## Configuration
Every option has a default and can be changed in the YAML file. Whenever the framework chooses
something itself (micro-batch, precision, validation source), it prints what it chose and why.

### Teacher (`configs/teacher_deepseek_flash.yaml`)
| Field | Default | Meaning |
|---|---|---|
| `base_url`, `model` | required | any OpenAI-compatible endpoint and model name |
| `api_key_env` | `DEEPSEEK_API_KEY` | environment variable holding the key (keys never go in config files) |
| `temperature` | 1.0 | > 0 so retries in rejection sampling give different answers (shipped config: 0.7) |
| `max_tokens` | 1024 | per call (shipped: 600) |
| `max_concurrency` | 32 | parallel requests (shipped: 256; price is per token, so more only finishes sooner) |
| `timeout_s`, `max_retries` | 300, 6 | per request; retries back off exponentially (not on 400/401/403/404) |
| `extra_body` | `{}` | provider-specific options, e.g. `thinking: {type: disabled}` for DeepSeek |
| `price` | 0 | `input_cache_hit`, `input_cache_miss`, `output` per 1M tokens, `peak_multiplier` |
| `peak` | none | `windows` (`[[start, end], ...]` hours), `utc_offset_hours`, `weekdays_only`; `teacher` refuses to start and stops cleanly inside these hours unless `--allow-peak` |
| `cache_path` | `cache/teacher.sqlite` | response cache: identical requests are never paid twice |

### SFT (`configs/sft_qwen3.yaml`)
| Field | Default | Meaning |
|---|---|---|
| `model`, `train_file`, `output_dir` | required | base model, JSONL of prompt/completion rows, where to write |
| `eval_file` | none | optional JSONL evaluated each epoch |
| `max_length` | 1024 | longer rows are **dropped**, never truncated (the length distribution is printed) |
| `epochs`, `lr` | 2, 1e-4 | 1e-4 suits LoRA; use ~1e-5 for full fine-tuning |
| `batch_size`, `grad_accum` | 8, 4 | per-device batch and accumulation (effective batch 32) |
| `warmup_ratio` | 0.03 | cosine schedule |
| `lora`, `lora_r`, `lora_alpha`, `lora_dropout` | true, 32, 64, 0.05 | LoRA on all linear layers; `lora: false` = full fine-tuning (fp32 master weights) |
| `seed`, `logging_steps`, `save_strategy` | 42, 10, `epoch` | |
| `extra` | `{}` | any other TRL `SFTConfig` argument; gradient checkpointing is on unless `extra: {gradient_checkpointing: false}` |

With LoRA, a merged copy is saved to `output_dir/merged` for evaluation and serving.

### GRPO (`configs/grpo_gsm8k_qwen3_0.6b.yaml`)
| Field | Default | Meaning |
|---|---|---|
| `model`, `output_dir` | required | start checkpoint (the SFT model) and where to write |
| `num_generations` | 8 | completions per prompt; the group is the baseline for the advantage |
| `max_prompt_length`, `max_completion_length` | 768, 256 | token limits |
| `temperature` | 1.0 | sampling temperature for rollouts |
| `lr` | 1e-6 | learning rate (full fine-tuning keeps fp32 master weights) |
| `beta` | 0.0 | KL coefficient to the start model (0 = no reference model) |
| `epsilon`, `loss_type` | 0.2, `dapo` | clipping and loss; `extra: {epsilon_high: 0.28}` for DAPO clip-higher |
| `batch_size`, `grad_accum` | 8, 4 | completions per device step and accumulation; the micro-batch shrinks automatically if long completions would not fit, keeping the product |
| `max_steps` | 500 | or set by `--epochs` |
| `use_vllm`, `vllm_mode`, `vllm_gpu_memory_utilization` | true, `colocate`, 0.3 | rollouts with vLLM on the training GPU |
| `lora` | false | true = LoRA (low-precision base) |
| `chat_template_kwargs` | `{}` | e.g. `{enable_thinking: false}`; prompts are rendered by the framework, so this works with any TRL version |
| `eval_every` | 25 | validation check every N steps (0 = off) |
| `eval_examples` | 200 | validation prompts: the task's `valid` split if it has one, else held out of training (at most 20%) |
| `eval_patience` | 0 | stop after N checks without improvement (0 = never stop early) |
| `entropy_floor_ratio`, `entropy_patience` | 0.1, 2 | stop when entropy stays below 10% of its starting value for 2 logs (0 = off) |
| `seed`, `logging_steps`, `save_steps` | 42, 5, 100 | |
| `extra` | `{}` | any other TRL `GRPOConfig` argument; checkpoints default to weights only, one kept (`save_only_model`, `save_total_limit`) |

## Outputs
| Command | Writes |
|---|---|
| `teacher` | `results/<task>/teacher_<split>_n<n>/`: `accepted.jsonl` (SFT rows), `log.jsonl` (every attempt and verdict; reruns resume from it), `stats.json` (yield, rejection reasons, tokens, cost) |
| `labels` | `data/sft/<task>/labels_<split>_n<n>.jsonl` |
| `sft` | `output_dir/` (adapter), `output_dir/merged/` (merged model), `sft_summary.json` |
| `eval` | `results/<task>/eval/<tag>/metrics.json` and `predictions.jsonl` (every completion and its score) |
| `grpo --filter-only` | `output_dir/pass_rate_filter.json` (statistics and advice), `output_dir/pass_rate_kept.txt` |
| `grpo` | `output_dir/` = **best model on validation** (a copy of the start model if nothing beat it), `output_dir/last/` = final model, `output_dir/validation.json` (scores per check, best step, weight change, entropy stop) |

## What the framework handles
- **Teacher client** (`distillops/teacher.py`): any OpenAI-compatible API; configurable concurrency,
  retries with backoff, an on-disk response cache (re-runs are free), a cost meter, and optional
  peak-price hours (refuses to start, and stops cleanly, inside them).
- **Adaptive rejection sampling** (`distillops/rejection.py`): retry only failed examples, up to
  `--max-attempts`; append-only logs so a killed run resumes where it stopped.
- **SFT** (`distillops/sft.py`): TRL, completion-only loss, over-long rows dropped (never truncated),
  LoRA, and a printout of exactly which tokens carry loss.
- **Evaluation** (`distillops/evaluate.py`): batched generation + `Task.score`; tasks can override decoding.
- **GRPO** (`distillops/grpo.py`): `Task.score` becomes the reward, so RL optimises the reported metric;
  pass-rate filter, validation-based checkpoint selection, entropy guard.
- **Hardware-agnostic** (`distillops/precision.py`, `memory.py`, `preflight.py`): precision chosen from
  the GPU, batch sizes from free memory and the model's KV-cache size, disk checked before training.

## Safeguards
Each one comes from a real failure; [docs/PITFALLS.md](docs/PITFALLS.md) has the details.

| Silent failure | Guard in the framework |
|---|---|
| bf16 weights round lr-1e-6 updates away: training curves rise but the model does not change | fp32 master weights for full fine-tuning; after training, the relative weight change is checked against a floor derived from lr and steps |
| Hard-coded bf16 breaks on GPUs without bf16 | `precision.compute_dtype()`: bf16, else fp16, else fp32, used everywhere |
| Windows spills VRAM to system RAM instead of raising OOM (10-100x slower) | batch sizes from free memory and the model's KV-cache size; halve automatically when a batch peaks above 85% |
| Loss-time logits grow with completion length (long math solutions ran out of memory, short titles did not) | GRPO micro-batch sized from completion length, vocabulary and memory, with accumulation raised to keep the effective batch |
| Checkpoints with optimizer state fill the disk at the final save | weights-only checkpoints, one kept; refuse to start if the disk cannot hold the run |
| Two jobs start vLLM on one GPU | wait for free memory before vLLM reserves its share |
| GRPO on prompts that are always right or always wrong learns nothing | pass-rate filter with plain-language advice; `--epochs` sizes the run to the kept prompts |
| GRPO entropy collapse: the policy memorises one answer per prompt | stop when entropy stays below 10% of the run's own starting value (relative, so no task-specific threshold) |
| Training reward rises while held-out quality falls | validation-based checkpoint selection; `output_dir` keeps the best model, or the start model if nothing beats it |

## Included examples and results
The two example tasks exist to show the workflow and what it achieves; their code lives in `tasks/`
and is not needed to use the framework on your own task.


### Math: GSM8K (Qwen3-0.6B student, all 1,319 test problems)
| Student | Accuracy |
|---|---|
| Untrained (step-by-step prompt) | 62.4% |
| SFT on answers only (no reasoning) | 13.6% |
| SFT on human-written GSM8K solutions | 49.0% |
| **SFT on DeepSeek reasoning (rejection-sampled)** | **62.5%** |
| **+ GRPO (1 epoch, reward = `Task.score`)** | **66.8%** |

Teacher: DeepSeek Flash, 7,286 of 7,473 problems accepted (97.5%), **3.76 CNY**. Reasoning is essential
(answers only: 13.6%) and the teacher's verified, explicit steps beat human solutions by 13.5 points.
SFT alone matches the untrained model (which its makers had already distilled on math); GRPO on top of
the SFT checkpoint adds **+4.3 points** (paired: p < 1e-5 vs SFT, p ≈ 1e-3 vs untrained). 64.7% of
training prompts carried a learning signal; token entropy went 0.35 -> 0.21 (no collapse).

### Recommendation: an LLM recommender as teacher (Amazon Industrial, 1,000 test users)
| Model | HR@10 | NDCG@10 |
|---|---|---|
| Popularity | .058 | .051 |
| SASRec (tuned) | .136 | .106 |
| Qwen3-0.6B, untrained | .103 | .061 |
| Qwen3-0.6B SFT on purchase histories | .138 | .097 |
| **Qwen3-1.7B SFT on purchase histories (teacher below)** | **.147** | **.1005** |

The LLM generates the next item's title under a catalog-trie constraint (no invalid items). It is
accurate but too large to serve, so it is used offline as the teacher for SASRec.

### Recommendation: distilling the LLM into deployable SASRec (full test set, 4,533 users, 3 seeds)
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

### Takeaways
1. For multi-step tasks, verified teacher reasoning is worth more than human-written data: +13.5 points
   on GSM8K, and reasoning is essential (answer-only training collapses to 13.6%). SFT on teacher data
   gives a clean starting point; GRPO with the same verifier as reward adds another +4.3 points.
2. Always compare against a same-data, no-teacher control (`python -m distillops labels`), not just the
   untrained model, to see what the teacher itself contributes.
3. For latency-bound products, use the LLM as an offline teacher: item embeddings plus soft labels lifted
   SASRec by 22-24% HR@10, and a two-stage recall + re-rank matched recent LLM recommenders, at
   unchanged serving cost.

### Reproduce the recommendation example
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
- **Distillation toolkits.** [EasyDistill](https://github.com/modelscope/easydistill) (Alibaba, EMNLP 2025
  demos) covers black-box data synthesis across text, vision-language, text-to-image and text-to-video,
  with outputs for LLaMA-Factory / ms-swift; [DistillKit](https://github.com/arcee-ai/DistillKit) focuses
  on logit-level distillation. distillops is narrower (text tasks) and goes deeper on the training loop:
  verification-based rejection sampling, no-teacher controls, built-in GRPO and its diagnostics.
  Training itself runs on [TRL](https://github.com/huggingface/trl).
- **LLM recommenders into sequential recommenders.** DLLM2Rec (RecSys 2024) distils LLM-based
  recommenders into conventional ones with ranking and embedding distillation, the approach followed
  here; LLM-ESR and LLMEmb use LLM item representations to strengthen sequential recommenders; SLMRec
  (ICLR 2025) distils into a smaller LLM. Benchmark and baselines follow
  [MiniOneRec](https://github.com/AkaliKong/MiniOneRec); S-GRec (arXiv 2602.10606) is the most recent
  result we compare against.

## Layout
```
distillops/       the framework: teacher, task interface, rejection sampling, sft, evaluate, grpo,
                  precision / memory / preflight, cli
tasks/            the two included example tasks (not needed for your own):
  math/           GSM8K, also the reference implementation of a Task
  rec/            recommendation: data, SASRec baseline and distillation, prompts, title trie,
                  constrained beams, item embeddings; resources/ holds the SFT user list
scripts/          check_task.py (any task); rec_*.py: the recommendation example (prepare, SASRec sweep,
                  build SFT data, LLM / SASRec eval, paired comparison, SASRec distillation, two-stage)
configs/          teacher, SFT and GRPO configs (example settings)
docs/PITFALLS.md  every silent failure met so far, its cause and the guard now in code
tests/            unit tests; no network or GPU needed
```
`data/`, `results/`, `checkpoints/` and `cache/` are created at run time and not tracked.

Data: GSM8K (MIT); MiniOneRec's preprocessed Amazon splits (Apache-2.0) of the Amazon Reviews dataset
(McAuley Lab). Code: MIT.
