"""Tests never touch the network: Hugging Face lookups stay offline."""

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
