"""Checks that fail fast, before hours of GPU time (see docs/PITFALLS.md).

    python -m distillops doctor      # environment report: GPU, precision, memory, disk, versions
"""

from __future__ import annotations

import importlib
import os
import platform
import shutil
from pathlib import Path
from typing import Any


def model_bytes(model: str, bytes_per_param: int = 2) -> int:
    """Size of a model's weights: its local weight files if present, else an estimate from
    its config (parameters x bytes_per_param)."""
    p = Path(model)
    if p.is_dir():
        files = list(p.glob("*.safetensors")) + list(p.glob("*.bin"))
        if files:
            return sum(f.stat().st_size for f in files)
    try:
        from transformers import AutoConfig

        from .memory import model_dims

        d = model_dims(AutoConfig.from_pretrained(model))
        layer = 12 * d["hidden"] ** 2  # attention + MLP, a standard rough count
        return int((d["layers"] * layer + 2 * d["vocab"] * d["hidden"]) * bytes_per_param)
    except Exception as e:  # unknown architecture or no network: disk checks are skipped
        print(f"note: could not estimate the size of {model} ({e}); disk-space checks are skipped", flush=True)
        return 0


def check_disk(out_dir: str | Path, need_bytes: int, what: str) -> None:
    """Stop before training if the output disk cannot hold what the run will write."""
    out = Path(out_dir)
    probe = out if out.exists() else next((q for q in out.parents if q.exists()), Path("."))
    free = shutil.disk_usage(probe).free
    if need_bytes and free < need_bytes * 1.1:
        raise SystemExit(f"Not enough disk for {what}: needs ~{need_bytes / 2**30:.1f} GB, "
                         f"{free / 2**30:.1f} GB free at {probe}. Free space or point output_dir elsewhere.")
    print(f"disk: {what} needs ~{need_bytes / 2**30:.1f} GB, {free / 2**30:.1f} GB free", flush=True)


def min_expected_change(lr: float, steps: int, weight_rms: float) -> float:
    """Lower bound on the relative weight change a working run should show. Adam moves each
    weight by about lr per step; even with random directions that is ~lr*sqrt(steps).
    We warn below a tenth of that (e.g. updates rounded away in low-precision weights)."""
    return 0.1 * lr * steps**0.5 / max(weight_rms, 1e-8)


def _version(mod: str) -> str:
    try:
        return getattr(importlib.import_module(mod), "__version__", "installed")
    except Exception:
        return "-"


def doctor() -> dict[str, Any]:
    from .precision import describe

    rep: dict[str, Any] = {"python": platform.python_version(), "os": platform.platform()}
    rep.update({m: _version(m) for m in ("torch", "transformers", "trl", "peft", "vllm", "datasets", "accelerate")})
    try:
        import torch

        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            rep["gpu"] = torch.cuda.get_device_name(0)
            rep["gpu_memory"] = f"{free / 2**30:.1f} GB free of {total / 2**30:.1f} GB"
        rep["precision"] = describe()
    except Exception as e:  # torch missing or broken
        rep["precision"] = f"torch unavailable: {e}"
    for label, path in (("disk (cwd)", "."), ("disk (HF cache)", os.environ.get("HF_HOME", Path.home() / ".cache"))):
        try:
            rep[label] = f"{shutil.disk_usage(path).free / 2**30:.1f} GB free"
        except OSError:
            pass
    notes = []
    if platform.system() == "Windows":
        notes.append("Windows (WDDM) does not raise out-of-memory: it spills VRAM to system RAM and runs "
                     "10-100x slower. Batch sizes are auto-limited; watch 'Shared GPU memory' in Task Manager.")
    if rep["vllm"] == "-":
        notes.append("vLLM not installed: GRPO generation and the pass-rate filter need it (Linux).")
    rep["notes"] = notes
    for k, v in rep.items():
        if k != "notes":
            print(f"{k:16s} {v}")
    for n in notes:
        print("note:", n)
    return rep
