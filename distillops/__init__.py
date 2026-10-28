"""Task-agnostic black-box distillation framework.

Core idea: a teacher LLM (API, no logits) produces outputs -> a task-specific
verifier keeps only outputs that agree with ground truth -> the kept outputs
become SFT data for a small student -> the same verifier later becomes the
GRPO reward.
"""
