# distillops-vision: distil an API model that sees images

An extension of [distillops](../../README.md) for tasks whose prompts contain images: document and
receipt extraction, chart questions, OCR, UI grounding, image classification, any task where a
program can check a text answer about an image. The workflow is distillops' own (teacher -> verify
-> SFT -> evaluation -> GRPO); only the inputs gain images. Nothing in distillops is changed.

> Status: run end to end on a real task with Qwen3-VL on one GPU (below); tested on CPU with tiny
> models of three architectures. Other architectures, GRPO with LoRA, and several GPUs have not been
> run on a GPU yet.

## Result: reading receipts
[vision_tasks/cord](vision_tasks/cord/README.md): a Qwen3-VL-2B student reads receipt photos into
JSON (items, totals, cash, change). Trained on the labels of Qwen3-VL-32B, accepted only when the
receipt's own arithmetic adds up (no human label), it scores **0.897** F1 on 200 test receipts, tying
its teacher (0.896) and up from 0.756 untrained. GRPO on 200 human-labelled receipts raises it to
0.905 and the share read perfectly from 42% to 49.5%; SFT on all 800 gold labels reaches 0.928.

## How it works
A vision-language model reads an image as extra input (tokens in the sequence, or features read
through cross-attention) and still writes text. So distillation is unchanged: the loss is on the
answer's tokens, the reward is `Task.score` on the answer's text, and gradients flow back through
whatever path the image took. What changes is everything that touches inputs: requests to the
teacher, cache keys, model loading, token counts, batching, memory estimates. That is what this
package adds.

Two choices keep runs sound:
- **The teacher sees what the student sees.** An image part carries its own `max_pixels`
  (`VisionTask.image()` writes the task's setting into it), and the teacher, the training rows,
  evaluation and GRPO all size it the same way. A teacher answering from detail the student cannot
  see would teach it to invent that detail.
- **Nothing architecture-specific is assumed.** Token counts are measured by encoding with the
  model's processor; vision modules are found by name (or named by you); tokens a text answer must
  never contain (image placeholders, image-generation tokens) are suppressed.

## Install
```bash
pip install -e ".[train,dev]"                 # distillops, from the repository root
pip install -e "extensions/vision[train,dev]"  # this extension (adds pillow, torchvision)
python -m distillops_vision doctor --model Qwen/Qwen2.5-VL-3B-Instruct
```

## Write a task
Subclass `VisionTask` and write it like any distillops `Task`; put images into messages with
`self.image(path)`:

```python
import json, re
from distillops.task import Verdict
from distillops_vision import VisionTask

class ReceiptTotal(VisionTask):
    name = "receipt_total"
    max_pixels = 1024 * 1024          # the size every component uses for these images

    def load_examples(self, split):   # rows {"id", "image": path, "total": "12.50"}
        return [json.loads(l) for l in open(f"data/receipts/{split}.jsonl")]

    def example_id(self, ex):
        return ex["id"]

    def teacher_messages(self, ex, attempt):
        return self.student_messages(ex)

    def student_messages(self, ex):
        return [{"role": "user", "content": [self.image(ex["image"]), {"type": "text",
                 "text": "What is the total on this receipt? End with: TOTAL: <amount>"}]}]

    def parse_teacher(self, ex, text):
        found = re.findall(r"TOTAL:\s*([\d.,]+)", text)
        return found[-1].replace(",", "") if found else None

    def verify(self, ex, parsed, text, attempt):
        if parsed is None:
            return Verdict(False, "parse_error")
        return Verdict(parsed == ex["total"], "ok" if parsed == ex["total"] else "wrong_total")

    def sft_example(self, ex, parsed, text):
        return {"prompt": self.student_messages(ex), "completion": [{"role": "assistant", "content": text}]}

    def score(self, ex, completion):
        return float(self.parse_teacher(ex, completion) == ex["total"])

    def gold_sft_example(self, ex):  # optional: the no-teacher control
        return {"prompt": self.student_messages(ex),
                "completion": [{"role": "assistant", "content": f"TOTAL: {ex['total']}"}]}
```

## Run
```bash
T=my_tasks.receipts:ReceiptTotal
S=Qwen/Qwen2.5-VL-3B-Instruct
python -m distillops_vision check-task --task $T --model $S          # rendered prompt, image tokens, loss mask
python -m distillops eval    --task $T --split test --model $S        # baseline (distillops' own command)
python -m distillops_vision teacher --task $T --teacher configs/my_vision_teacher.yaml --n 100
python -m distillops_vision teacher --task $T --teacher configs/my_vision_teacher.yaml
python -m distillops_vision sft extensions/vision/configs/sft_vlm.yaml \
         --train_file results/receipt_total/teacher_train_n0/accepted.jsonl --output_dir checkpoints/receipts_sft
python -m distillops eval    --task $T --split test --model checkpoints/receipts_sft/merged
python -m distillops labels  --task $T                                 # no-teacher control rows
G="extensions/vision/configs/grpo_vlm.yaml --model checkpoints/receipts_sft/merged --output_dir checkpoints/receipts_grpo"
python -m distillops_vision grpo $G --task $T --filter-samples 8 --filter-only
python -m distillops_vision grpo $G --task $T --keep-ids checkpoints/receipts_grpo/pass_rate_kept.txt --epochs 1
python -m distillops eval    --task $T --split test --model checkpoints/receipts_grpo
python -m distillops_vision usage --teacher configs/my_vision_teacher.yaml
```
`labels` and `eval` are distillops' commands, unchanged. The teacher config is distillops' plus
`max_pixels` and `image_detail` (`configs/teacher_vision_template.yaml`); the SFT and GRPO configs
are distillops' plus:

| Field | Default | Meaning |
|---|---|---|
| `train_vision` | false | train the image encoder too (more memory; rarely needed). With LoRA it trains in full next to the adapters |
| `train_projector` | true | train the layers that map image features into the language model (in fp32, also next to LoRA) |
| `vision_modules` | none | regexes over module paths, to name the vision modules by hand |
| `trust_remote_code` | false | run the model repository's own code (a few models need it) |

Image size is set once, by the task (`VisionTask.max_pixels`), and travels with each image, so
there is no size setting in these configs. `max_length` (SFT) and `max_prompt_length` (GRPO) count
image tokens: longer rows are dropped and longer prompts left out, never truncated through their
images. A task may mix prompts with and without images, and prompts with several images.

## Architectures
| Architecture | How the image enters | Tested (tiny models, CPU) |
|---|---|---|
| ViT + projector in the sequence (Qwen2-VL / Qwen2.5-VL / Qwen3-VL, InternVL, Gemma 3, SmolVLM) | image tokens in the sequence | Qwen2-VL: everything, including mixed and multi-image prompts. LLaVA: everything; GRPO only with exactly one image per prompt (TRL splits its per-image pixel values one row per prompt; other cases stop before training and say so) |
| ViT + cross-attention (Llama 3.2 Vision) | one image token; features read by cross-attention | Mllama: SFT, eval (a task mixing image and text-only prompts needs `batch_size: 1` in SFT; you are told). GRPO: TRL 0.24 does not pass its image inputs, so it stops before training and says so |
| Natively multimodal with an encoder (Llama 4) | as the first row | not tested |
| Discrete image tokens (Chameleon, Emu3) | image tokens in the vocabulary | not tested; image-generation tokens are suppressed when answering |
| Encoder-free (Fuyu) | raw patches | not tested; `train_vision` does not apply (no encoder to freeze) |

GRPO with images needs TRL >= 0.20 and < 0.28 (0.28 removed APIs it uses), and so transformers < 5
(older TRL does not import with it); the `train` extra installs both; with vLLM rollouts, a vLLM that serves the architecture
(`doctor` reports both). With vLLM colocated on the training GPU, the trained weights go to vLLM in
one call per step (frozen ones are not sent), validation generates with that vLLM, images stay
decoded in memory, and CPU preprocessing uses the container's CPU quota: on the receipts task this
took a step from about 130 s to 45 s. Other setups, or TRL/vLLM versions without the internals this
uses, fall back to TRL's own weight sync (tested with TRL 0.24, vLLM 0.11). Rollouts never sample image placeholder tokens (`suppress_tokens` with
transformers, a logit bias with vLLM). The vision extension's own SFT uses transformers' Trainer, so it does
not depend on how a TRL version handles vision models.

## What it uses from distillops
The extension subclasses `Task`, `Teacher`, `TeacherConfig`, `SFTRunConfig` and `GRPORunConfig`, and
calls distillops for rejection sampling, run fingerprints, evaluation and its files, the pass-rate
filter's statistics, validation selection, the entropy guard, stop tokens, trained-id records,
precision, memory and disk checks. `tests/test_contract_cli.py` lists every name and parameter it
relies on, so a change in distillops that would break the extension fails a test first.

## Layout
```
distillops_vision/
  images.py     image parts, sizing (max_pixels), content hashes, data URLs
  task.py       VisionTask (evaluation through distillops' evaluate_task)
  teacher.py    VisionTeacher: image_url requests, cache keys by image content and size
  model.py      loading any vision-language model, rendering, token counts, vision modules
  generate.py   batched greedy generation (evaluation, GRPO validation)
  sft.py        vision SFT (answer-only loss, frozen encoder, LoRA on the language model)
  grpo.py       vision GRPO and the vLLM pass-rate filter
  compat.py     library support checks; doctor's vision report
  cli.py        teacher | sft | grpo | usage | merge | doctor | check-task
configs/        teacher, SFT and GRPO templates
vision_tasks/cord/  the receipts example: task, data preparation, GRPO pools, paired comparison
tests/          CPU tests with tiny Qwen2-VL, LLaVA and Mllama models built offline
```
