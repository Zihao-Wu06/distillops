"""Checks that fail fast, before hours of GPU time.

    python -m distillops doctor      # environment report: GPU, precision, memory, disk, versions
"""

from __future__ import annotations

import importlib
import json
import math
import os
import platform
import shutil
import struct
from pathlib import Path
from typing import Any


def _weight_files(d: Path) -> list[Path] | None:
    """The safetensors files of a model folder (from its index when sharded); None if incomplete."""
    index = d / "model.safetensors.index.json"
    if index.is_file():
        names = sorted(set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values()))
        files = [d / n for n in names]
        return files if all(f.is_file() for f in files) else None
    files = sorted(d.glob("*.safetensors"))
    return files or None


def _safetensors_params(files: list[Path]) -> int:
    """Exact parameter count from safetensors headers (8-byte length + JSON), whatever the dtype."""
    total = 0
    for f in files:
        with open(f, "rb") as fh:
            header = json.loads(fh.read(struct.unpack("<Q", fh.read(8))[0]))
        total += sum(math.prod(v["shape"]) for k, v in header.items() if k != "__metadata__")
    return total


def _local_params(model: str) -> int | None:
    """From weight files on disk: a model folder, or the Hugging Face cache (no download)."""
    p = Path(model)
    folders = [p] if p.is_dir() else []
    if not folders:
        try:
            from huggingface_hub import snapshot_download

            folders.append(Path(snapshot_download(model, local_files_only=True)))
        except Exception:  # not cached, or not a hub id
            return None
    for d in folders:
        files = _weight_files(d)
        if files:
            return _safetensors_params(files)
        bins = sorted(d.glob("*.bin"))
        if bins:  # PyTorch pickles: size / bytes per parameter of the stored dtype
            cfg = json.loads((d / "config.json").read_text(encoding="utf-8")) if (d / "config.json").is_file() else {}
            width = 4 if str(cfg.get("torch_dtype") or cfg.get("dtype")) == "float32" else 2
            return sum(f.stat().st_size for f in bins) // width
    return None


def _hub_params(model: str) -> int | None:
    """From the Hub's safetensors metadata (a small request; skipped offline)."""
    if os.environ.get("HF_HUB_OFFLINE", "").lower() in ("1", "true", "yes"):
        return None
    try:
        from huggingface_hub import get_safetensors_metadata

        return sum(get_safetensors_metadata(model).parameter_count.values()) or None
    except Exception:
        return None


_GATED = ("silu", "swish", "swiglu", "geglu", "gelu_pytorch_tanh")  # activations of gated (3-matrix) MLPs


def estimate_params(config: Any) -> int:
    """Rough parameter count from a config: attention (with grouped KV heads), MLP (gated or not,
    mixture-of-experts aware) and embeddings. Used only when no weight files or metadata exist."""
    from .memory import model_dims

    d = model_dims(config)
    get_text = getattr(config, "get_text_config", None)
    cfg = get_text() if callable(get_text) else getattr(config, "text_config", config)

    def first(*names: str) -> Any:
        return next((getattr(cfg, n) for n in names if isinstance(getattr(cfg, n, None), int) and getattr(cfg, n) > 0),
                    None)

    h = d["hidden"]
    attn = 2 * h * d["heads"] * d["head_dim"] + 2 * h * d["kv_heads"] * d["head_dim"]
    act = str(getattr(cfg, "hidden_act", None) or getattr(cfg, "hidden_activation", None)
              or getattr(cfg, "activation_function", "")).lower()
    mats = 3 if any(g in act for g in _GATED) else 2
    inter = first("intermediate_size", "n_inner", "ffn_hidden_size", "ffn_dim") or 4 * h
    experts = first("num_experts", "num_local_experts", "n_routed_experts")
    if experts and experts > 1:
        mlp = experts * mats * h * (first("moe_intermediate_size") or inter)
        mlp += mats * h * (first("shared_expert_intermediate_size") or 0)
    else:
        mlp = mats * h * inter
    tied = bool(getattr(cfg, "tie_word_embeddings", getattr(config, "tie_word_embeddings", True)))
    return d["layers"] * (attn + mlp) + d["vocab"] * h * (1 if tied else 2)


def param_count(model: str) -> int:
    """Parameters of a model (hub id or folder): exact from weight files or Hub metadata when
    available, else estimated from its config; 0 if none of these work (checks are then skipped)."""
    n = _local_params(model) or _hub_params(model)
    if n:
        return n
    try:
        from transformers import AutoConfig

        return estimate_params(AutoConfig.from_pretrained(model))
    except Exception as e:  # unknown architecture or no network
        print(f"note: could not estimate the size of {model} ({e}); size-based checks are skipped", flush=True)
        return 0


def stored_bytes(model: str) -> int:
    """Bytes of a local model folder's weight files (what copying it costs); 0 if it is not local."""
    p = Path(model)
    if not p.is_dir():
        return 0
    files = _weight_files(p) or sorted(p.glob("*.bin"))
    return sum(f.stat().st_size for f in files)


def model_bytes(model: str, bytes_per_param: int = 2) -> int:
    """Size of a model's weights at `bytes_per_param` (2 = bf16/fp16, 4 = fp32)."""
    return param_count(model) * bytes_per_param


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
