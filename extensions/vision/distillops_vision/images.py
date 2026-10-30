"""Images in chat messages: the part format, loading, resizing and content hashes.

A task puts an image into a message as {"type": "image", "path": "data/x/0001.png"} (optionally
with "max_pixels": N); a message's content is then a list of parts (images and
{"type": "text", "text": ...}) instead of a string. Rows on disk keep the path, never the pixels.

An image part carries its own size limit (VisionTask.image writes the task's max_pixels into it),
so training rows, evaluation and GRPO size every image the same way without repeating the setting,
and the teacher sees what the student sees. Every component sizes images with `fit`.
"""

from __future__ import annotations

import base64
import hashlib
import io
import math
from pathlib import Path
from typing import Any, Iterable


def is_image(part: Any) -> bool:
    return isinstance(part, dict) and part.get("type") == "image"


def image_parts(messages: Iterable[dict]) -> list[dict]:
    """The image parts of `messages`, in order."""
    out = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            for part in content:
                if is_image(part):
                    if "path" not in part:
                        raise ValueError(f"an image part needs a 'path' (got {part})")
                    out.append(part)
    return out


def image_paths(messages: Iterable[dict]) -> list[str]:
    return [str(p["path"]) for p in image_parts(messages)]


def has_images(messages: Iterable[dict]) -> bool:
    return bool(image_parts(messages))


def pixels(part: dict, default: int | None = None) -> int | None:
    """The size limit for an image part: its own "max_pixels", else `default`."""
    value = part.get("max_pixels")
    return int(value) if value else default


def fit(size: tuple[int, int], max_pixels: int | None) -> tuple[int, int]:
    """(width, height) scaled down, keeping the aspect ratio, so width * height <= max_pixels."""
    w, h = size
    if not max_pixels or w * h <= max_pixels:
        return w, h
    scale = math.sqrt(max_pixels / (w * h))
    return max(1, int(w * scale)), max(1, int(h * scale))


def load_part(part: dict, default_max_pixels: int | None = None):
    return load(part["path"], pixels(part, default_max_pixels))


def load(path: str | Path, max_pixels: int | None = None):
    """The image as RGB, scaled down to at most `max_pixels`."""
    from PIL import Image

    with Image.open(path) as im:
        im = im.convert("RGB")
        size = fit(im.size, max_pixels)
        return im if size == im.size else im.resize(size, Image.Resampling.BICUBIC)


def content_hash(path: str | Path) -> str:
    """sha256 of the file's bytes. Not cached: a file rewritten in place can keep its size and
    modification time (coarse timestamps), and hashing an image takes about a millisecond."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def to_data_url(path: str | Path, max_pixels: int | None = None) -> str:
    """The image, sized as for the student, as a data URL for an OpenAI-compatible API.
    JPEG sources stay JPEG (smaller requests); everything else is sent as PNG."""
    im = load(path, max_pixels)
    jpeg = Path(path).suffix.lower() in (".jpg", ".jpeg")
    buf = io.BytesIO()
    im.save(buf, format="JPEG" if jpeg else "PNG", **({"quality": 95} if jpeg else {}))
    mime = "image/jpeg" if jpeg else "image/png"
    return f"data:{mime};base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"


def for_template(messages: list[dict]) -> list[dict]:
    """Messages in the form processors' chat templates expect: every content a list of parts, an
    image part reduced to {"type": "image"} (the pixels go to the processor separately)."""
    out = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, str):
            parts = [{"type": "text", "text": content}]
        else:
            parts = [{"type": "image"} if is_image(p) else p for p in content or []]
        out.append({**m, "content": parts})
    return out
