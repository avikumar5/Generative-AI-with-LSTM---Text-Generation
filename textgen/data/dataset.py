"""Sliding-window sequences for next-word prediction."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, random_split

from textgen.config import BATCH_SIZE, SEQ_LENGTH, VAL_SPLIT
from textgen.data.preprocess import Vocab, tokenize


class TextSequenceDataset(Dataset):
    """
    Each item is (context of length seq_len, next-token target).
    Built once from a flat list of word ids.
    """

    def __init__(self, ids: list[int], seq_len: int = SEQ_LENGTH):
        if len(ids) <= seq_len:
            raise ValueError(
                f"Need more than {seq_len} tokens, got {len(ids)}"
            )
        self.ids = ids
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.ids) - self.seq_len

    def __getitem__(self, idx: int):
        chunk = self.ids[idx : idx + self.seq_len + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[-1], dtype=torch.long)
        return x, y


def load_token_ids(
    path: Path,
    min_freq: int = 2,
    max_tokens: int | None = None,
) -> tuple[list[int], Vocab]:
    raw = path.read_text(encoding="utf-8")
    tokens = tokenize(raw)
    if max_tokens is not None:
        tokens = tokens[:max_tokens]
    vocab = Vocab(tokens, min_freq=min_freq)
    ids = vocab.encode(tokens)
    return ids, vocab


def build_dataloaders(
    path: Path,
    seq_len: int = SEQ_LENGTH,
    batch_size: int = BATCH_SIZE,
    val_split: float = VAL_SPLIT,
    min_freq: int = 2,
    max_tokens: int | None = None,
) -> tuple[DataLoader, DataLoader, Vocab]:
    ids, vocab = load_token_ids(path, min_freq=min_freq, max_tokens=max_tokens)
    full = TextSequenceDataset(ids, seq_len=seq_len)

    n_val = max(1, int(len(full) * val_split))
    n_train = len(full) - n_val
    train_ds, val_ds = random_split(
        full,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, drop_last=False
    )
    return train_loader, val_loader, vocab
