"""LSTM language model — embedding → LSTM stack → linear over vocab."""

from __future__ import annotations

import torch
import torch.nn as nn


class LSTMTextGenerator(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def forward(self, x, hidden=None):
        # x: (B, T)
        emb = self.drop(self.embed(x))
        out, hidden = self.lstm(emb, hidden)
        # next-token from the last timestep
        logits = self.fc(self.drop(out[:, -1, :]))
        return logits, hidden

    def init_hidden(self, batch_size: int, device: torch.device):
        h = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        return h, c
