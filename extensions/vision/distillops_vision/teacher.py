"""A teacher that sees images: any OpenAI-compatible chat API with image input.

The framework's Teacher does the work (concurrency, retries, the response cache, token counts);
this subclass changes two things:
- requests: image parts become OpenAI `image_url` parts (base64 data URLs), sized with
  `max_pixels` exactly as the student sees them;
- cache keys: an image counts by the sha256 of its file and the size it is sent at, not by its
  path or pixels, so moving the data keeps the cache and a changed image is asked again.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from distillops.teacher import Teacher, TeacherConfig

from . import images


@dataclass
class VisionTeacherConfig(TeacherConfig):
    # Largest image sent (width x height). None = each image part's own size (VisionTask.max_pixels),
    # so the teacher sees what the student will see; set it only to send the teacher something else.
    max_pixels: int | None = None
    # OpenAI-style image detail ("low", "high", "auto") for providers that accept it; None = not sent
    image_detail: str | None = None

    def identity(self) -> dict[str, Any]:
        return {**super().identity(), "max_pixels": self.max_pixels, "image_detail": self.image_detail}


class VisionTeacher(Teacher):
    def __init__(self, cfg: TeacherConfig, client: Any | None = None, max_pixels: int | None = None) -> None:
        super().__init__(cfg, client)
        # an override for every image; None = each part's own max_pixels
        self.max_pixels = max_pixels if max_pixels is not None else getattr(cfg, "max_pixels", None)
        self.detail = getattr(cfg, "image_detail", None)

    def _pixels(self, part: dict) -> int | None:
        return self.max_pixels if self.max_pixels is not None else images.pixels(part)

    def _for_key(self, messages: list[dict]) -> list[dict]:
        out = []
        for m in messages:
            content = m.get("content")
            if isinstance(content, list):
                content = [{"type": "image", "sha256": images.content_hash(p["path"]), "max_pixels": self._pixels(p),
                            "detail": self.detail} if images.is_image(p) else p for p in content]
            out.append({**m, "content": content})
        return out

    def _encode(self, messages: list[dict]) -> list[dict]:
        out = []
        for m in messages:
            content = m.get("content")
            if isinstance(content, list):
                parts = []
                for p in content:
                    if images.is_image(p):
                        url = {"url": images.to_data_url(p["path"], self._pixels(p))}
                        if self.detail:
                            url["detail"] = self.detail
                        parts.append({"type": "image_url", "image_url": url})
                    else:
                        parts.append(p)
                content = parts
            out.append({**m, "content": content})
        return out

    def _key(self, messages: list[dict], sample_id: int, params: dict, with_endpoint: bool = True) -> str:
        return super()._key(self._for_key(messages), sample_id, params, with_endpoint)

    async def _call_with_retry(self, messages: list[dict], params: dict, extra: dict) -> Any:
        return await super()._call_with_retry(self._encode(messages), params, extra)


def image_fingerprint(task: Any, examples: list[Any]) -> str | None:
    """Hash of the images in the teacher prompt of the split's example with the smallest id (the
    same example distillops' task_fingerprint hashes the prompt text of): a run cannot be resumed
    after its images were replaced under the same paths."""
    import hashlib

    from distillops.task import check_ids

    if not examples:
        return None
    ids = check_ids(task, examples)
    probe = examples[min(range(len(ids)), key=ids.__getitem__)]
    hashes = [images.content_hash(p) for p in images.image_paths(task.teacher_messages(probe, 0))]
    return hashlib.sha256("|".join(hashes).encode()).hexdigest()[:16] if hashes else None
