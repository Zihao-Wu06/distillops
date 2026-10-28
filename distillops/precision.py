"""Numeric precision in one place, chosen from the hardware instead of hard-coded.

Rules (each one learned the hard way, see docs/PITFALLS.md):
- Compute dtype: bf16 if the GPU supports it (Ampere and newer), else fp16, else fp32 (CPU).
- Full fine-tuning keeps fp32 *master* weights and computes in the low-precision dtype
  (mixed precision). Training bf16 weights directly loses small updates: with lr 1e-6 an
  update is ~1e-4 of a weight, below bf16's ~4e-3 resolution, so it rounds away.
- LoRA keeps the frozen base in the compute dtype; PEFT keeps the adapter weights in fp32.
- Inference (evaluation, vLLM sampling) uses the compute dtype.
"""

from __future__ import annotations

from typing import Any


def _torch():
    import torch

    return torch


def bf16_supported() -> bool:
    torch = _torch()
    return bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())


def compute_dtype():
    """Low-precision dtype for matmuls and inference on this machine."""
    torch = _torch()
    if not torch.cuda.is_available():
        return torch.float32
    return torch.bfloat16 if bf16_supported() else torch.float16


def dtype_name(dtype: Any) -> str:
    return str(dtype).replace("torch.", "")


def train_settings(full_finetune: bool) -> dict[str, Any]:
    """Arguments for HF/TRL trainers: weight dtype at load time plus the mixed-precision flag."""
    torch = _torch()
    comp = compute_dtype()
    return {
        # trainable weights must be fp32 for full fine-tuning; a frozen LoRA base can stay low precision
        "model_dtype": torch.float32 if (full_finetune or comp == torch.float32) else comp,
        "bf16": comp == torch.bfloat16,
        "fp16": comp == torch.float16,
    }


def describe() -> str:
    torch = _torch()
    if not torch.cuda.is_available():
        return "no CUDA GPU: fp32 on CPU"
    name = torch.cuda.get_device_name(0)
    comp = dtype_name(compute_dtype())
    return (f"{name}: compute {comp}; full fine-tuning = fp32 master weights + {comp} autocast; "
            f"LoRA = {comp} base + fp32 adapters")
