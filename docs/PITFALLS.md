# Pitfalls and the guards built for them

Most of these failures raise no error: the run finishes, the curves look normal, and the result is
wrong or 10-100x slower. Each one now has an automatic check in the framework. Run
`python -m distillops doctor` on a new machine first.

## Precision

### 1. bf16 weights round small updates away
- **Symptom.** A 151-step GRPO run raised the training reward from 0.40 to 0.48, but test metrics were
  identical to the starting model and only 10 of 1,000 predictions changed.
- **Cause.** The model was loaded and updated in bf16. bf16 keeps about 3 significant digits, so the
  smallest representable change is ~0.4% of a value; with lr 1e-6 an update is ~0.005% of a weight and
  is rounded away. Measured mean relative weight change: 4e-5; 54 of 310 weight matrices were unchanged.
  The rising reward was noise on a small prompt set.
- **Guard.** `distillops/precision.py`: full fine-tuning always keeps fp32 master weights and computes in
  bf16/fp16 (mixed precision); a frozen LoRA base stays in low precision while PEFT keeps adapters in
  fp32. After training, the relative weight change is compared with a floor derived from the learning
  rate and step count (`preflight.min_expected_change`) and a warning is printed below it.

### 2. Hard-coded precision
- **Symptom.** `torch.bfloat16` written in a dozen places fails or overflows on GPUs without bf16.
- **Guard.** `precision.compute_dtype()` picks bf16 if supported, else fp16, else fp32 (CPU). Training,
  evaluation, vLLM sampling and embedding extraction all take their dtype from it.

## Memory

### 3. Windows spills VRAM to system RAM instead of raising out-of-memory
- **Symptom.** Evaluation went from 0.7 to 105 s per example; GPU utilisation read 99% at ~120 W.
- **Cause.** The Windows display driver (WDDM) pages GPU memory to system RAM when the card is full,
  without an error.
- **Guard.** Generation batch sizes are computed from free memory and the model's KV-cache size, using
  70% of what is free; after each batch, if peak memory exceeded 85% of the card, the next batch is
  halved (`memory.adaptive_chunks`); a real out-of-memory error halves the batch and retries.
  `doctor` warns about this on Windows.

### 4. The KV-cache estimate is half the real peak
- **Cause.** The cache grows token by token and is reallocated as it grows, so peak usage is ~2x the
  final cache; beam search also copies the cache when it reorders beams (~3.7x measured).
- **Guard.** The estimate uses these measured factors, and the adaptive halving above catches the rest.

### 5. Cached-but-unused memory looks occupied
- **Symptom.** Right after one phase, the next phase sized its batch to 1.
- **Cause.** The driver reports memory that PyTorch has cached but is not using as taken.
- **Guard.** Free memory includes `memory_reserved - memory_allocated`; validation during GRPO also
  releases the training step's cache before sizing its batch.

### 6. Loss-time logits grow with completion length
- **Symptom.** GRPO on GSM8K ran out of memory at step 1; the same settings worked for recommendation.
- **Cause.** The loss needs fp32 logits over the whole vocabulary for every completion token: 64
  completions of 512 tokens need ~20 GB; 64 product titles of ~30 tokens need ~4 GB.
- **Guard.** `grpo.fit_micro_batch` shrinks the per-device batch from completion length, vocabulary
  size and memory, and raises gradient accumulation so completions per step stay the same.

### 7. Two jobs start vLLM on one GPU
- **Guard.** `memory.wait_for_free_memory` waits until vLLM's share is free before it is reserved.

## Disk

### 8. Checkpoints fill the disk at the final save
- **Symptom.** A 1.7B full fine-tuning GRPO run finished all steps and failed writing the final model.
- **Cause.** Each checkpoint stored optimizer state as well as weights (~10-17 GB each).
- **Guard.** GRPO checkpoints are weights only and one is kept by default; before training, the run's
  expected writes are estimated from the model size and the run refuses to start if the disk is too
  small (`preflight.check_disk`).

## Libraries

### 9. Version combinations that do not fit
- **Symptom.** pip resolved transformers 5.x next to vLLM 0.11 (released before transformers 5); TRL 0.24
  has no `chat_template_kwargs`; transformers 4.x and 5.x spell the warmup ratio differently.
- **Guard.** A tested pin for GRPO (vLLM 0.11 + transformers 4.57.1 + TRL 0.24); chat-template options
  are rendered by the framework, independent of the TRL version; warmup is passed in the form the
  installed transformers expects. `doctor` lists every version.

## Reinforcement learning

### 10. GRPO on prompts that are always right or always wrong learns nothing
- With all completions equally rewarded, the group-normalised advantage is zero. On recommendation only
  5.4% of prompts were "sometimes right"; on GSM8K, 64.7%.
- **Guard.** `grpo --filter-samples 8` measures this first, keeps only the useful prompts and prints
  advice (`grpo.diagnose_pass_rates`); `--epochs` sizes the run to the number of kept prompts.

### 11. Entropy collapse
- **Symptom.** Training reward rose from 0.41 to 0.93 within ~65 steps while token entropy fell from 0.87
  to 0.002: the policy repeated one memorised answer per training prompt.
- **Guard.** Training stops when entropy stays below `entropy_floor_ratio` (10%) of its own starting
  value for `entropy_patience` logs. The threshold is relative because entropy scales differ by task.

### 12. Training reward rises while held-out quality falls
- **Symptom.** The collapsed run above lowered HR@20 on new users from .172 to .162; a KL-regularised run
  stopped the drop but did not improve on the start.
- **Guard.** Validation-based checkpoint selection: every `eval_every` steps, greedy generation on
  `eval_examples` prompts that are not trained on, scored with `Task.score`. `output_dir` keeps the best
  model; if no trained step beats the start, the start model is delivered and the log says so.

## Experiment design

### 13. Comparing only with the untrained model overstates the teacher
- In one experiment, an untrained model at 0.103 rose to 0.124 after training on teacher reasoning, which
  looked like a 20% gain; training on the plain labels of the same examples reached 0.138. Compare
  against a same-data, no-teacher control (`python -m distillops labels`) to see what the teacher adds.

### 14. Teachers break instructions in ways that teach the wrong thing
- A teacher told not to mention information the student will never see did so in about 12% of its
  answers. `Task.verify` is where such outputs are rejected; read accepted samples before scaling up.

### 15. A dependency the step does not need
- Loading training examples also read a file needed only to build teacher prompts, so a later stage
  failed on a machine without it. Tasks now read such files only where they are needed and fail with a
  message that says how to create them.
