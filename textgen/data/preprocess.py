"""Text cleaning and word-level vocabulary."""

from __future__ import annotations

import re
import string
from collections import Counter


_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def clean_text(text: str) -> str:
    """Lowercase, drop punctuation, collapse whitespace."""
    text = text.lower()
    # Gutenberg / OCR leftovers
    text = (
        text.replace("\u2019", "'")
        .replace("\u2018", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2014", " ")
        .replace("\u2013", " ")
    )
    text = text.translate(_PUNCT_TABLE)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text: str) -> list[str]:
    return clean_text(text).split()


class Vocab:
    """word ↔ index mapping. Unknown tokens collapse to <unk>."""

    PAD = "<pad>"
    UNK = "<unk>"

    def __init__(self, tokens: list[str], min_freq: int = 2):
        counts = Counter(tokens)
        words = [w for w, c in counts.most_common() if c >= min_freq]

        self.itos = [self.PAD, self.UNK] + words
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, tokens: list[str]) -> list[int]:
        unk = self.stoi[self.UNK]
        return [self.stoi.get(t, unk) for t in tokens]

    def decode(self, indices: list[int]) -> str:
        return " ".join(self.itos[i] for i in indices if i < len(self.itos))

    def state_dict(self) -> dict:
        return {"itos": self.itos}

    @classmethod
    def from_state_dict(cls, state: dict) -> "Vocab":
        v = object.__new__(cls)
        v.itos = list(state["itos"])
        v.stoi = {w: i for i, w in enumerate(v.itos)}
        return v
