"""distillops-vision: the distillops workflow for tasks whose prompts contain images.

An extension: it changes nothing in distillops and builds on its public pieces (see README.md,
"What it uses from distillops"). Write a task by subclassing `VisionTask`.
"""

from .task import VisionTask

__all__ = ["VisionTask"]
