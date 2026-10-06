"""
Runtime settings. Prefer a local .env (see .env.example); otherwise defaults kick in.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def _int(key: str, default: int) -> int:
    return int(os.getenv(key, default))


def _float(key: str, default: float) -> float:
    return float(os.getenv(key, default))


def _path(key: str, default: Path) -> Path:
    raw = os.getenv(key)
    return Path(raw) if raw else default


DATA_PATH = _path("DATA_PATH", ROOT / "data" / "shakespeare.txt")
CHECKPOINT_DIR = _path("CHECKPOINT_DIR", ROOT / "checkpoints")
SAMPLE_DIR = _path("SAMPLE_DIR", ROOT / "samples")

SEQ_LENGTH = _int("SEQ_LENGTH", 40)
BATCH_SIZE = _int("BATCH_SIZE", 64)
EMBED_DIM = _int("EMBED_DIM", 128)
HIDDEN_DIM = _int("HIDDEN_DIM", 256)
NUM_LAYERS = _int("NUM_LAYERS", 2)
DROPOUT = _float("DROPOUT", 0.3)
EPOCHS = _int("EPOCHS", 15)
LEARNING_RATE = _float("LEARNING_RATE", 0.002)
VAL_SPLIT = _float("VAL_SPLIT", 0.1)
EARLY_STOP_PATIENCE = _int("EARLY_STOP_PATIENCE", 3)

TEMPERATURE = _float("TEMPERATURE", 0.8)
GEN_LENGTH = _int("GEN_LENGTH", 200)
SEED_TEXT = os.getenv("SEED_TEXT", "to be or not to be")

# Project Gutenberg — Complete Works of William Shakespeare (plain text)
GUTENBERG_URL = "https://www.gutenberg.org/files/100/100-0.txt"
