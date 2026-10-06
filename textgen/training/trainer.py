"""Training loop with validation, early stopping, and checkpoints."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from textgen.config import (
    CHECKPOINT_DIR,
    EARLY_STOP_PATIENCE,
    EPOCHS,
    LEARNING_RATE,
)
from textgen.data.preprocess import Vocab
from textgen.model.lstm import LSTMTextGenerator


def save_checkpoint(
    path: Path,
    model: LSTMTextGenerator,
    vocab: Vocab,
    optimizer: torch.optim.Optimizer | None = None,
    epoch: int = 0,
    val_loss: float = float("inf"),
    meta: dict | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "vocab": vocab.state_dict(),
        "epoch": epoch,
        "val_loss": val_loss,
        "meta": meta or {},
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    torch.save(payload, path)


def load_checkpoint(
    path: Path,
    device: torch.device | None = None,
) -> tuple[LSTMTextGenerator, Vocab, dict]:
    device = device or torch.device("cpu")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    vocab = Vocab.from_state_dict(ckpt["vocab"])
    meta = ckpt.get("meta", {})

    model = LSTMTextGenerator(
        vocab_size=len(vocab),
        embed_dim=meta.get("embed_dim", 128),
        hidden_dim=meta.get("hidden_dim", 256),
        num_layers=meta.get("num_layers", 2),
        dropout=meta.get("dropout", 0.3),
    )
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model, vocab, ckpt


class Trainer:
    def __init__(
        self,
        model: LSTMTextGenerator,
        train_loader: DataLoader,
        val_loader: DataLoader,
        vocab: Vocab,
        device: torch.device | None = None,
        lr: float = LEARNING_RATE,
        checkpoint_dir: Path = CHECKPOINT_DIR,
        patience: int = EARLY_STOP_PATIENCE,
        meta: dict | None = None,
    ):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.vocab = vocab
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.patience = patience
        self.meta = meta or {}
        self.history: list[dict] = []

    def _run_epoch(self, loader: DataLoader, train: bool) -> float:
        if train:
            self.model.train()
        else:
            self.model.eval()

        total, n = 0.0, 0
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for x, y in tqdm(loader, leave=False, desc="train" if train else "val"):
                x, y = x.to(self.device), y.to(self.device)
                if train:
                    self.optimizer.zero_grad(set_to_none=True)

                logits, _ = self.model(x)
                loss = self.criterion(logits, y)

                if train:
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()

                total += loss.item() * x.size(0)
                n += x.size(0)

        return total / max(n, 1)

    def fit(self, epochs: int = EPOCHS) -> Path:
        best_val = float("inf")
        bad_epochs = 0
        best_path = self.checkpoint_dir / "best.pt"

        print(f"device={self.device}  vocab={len(self.vocab)}  epochs={epochs}")

        for epoch in range(1, epochs + 1):
            train_loss = self._run_epoch(self.train_loader, train=True)
            val_loss = self._run_epoch(self.val_loader, train=False)
            self.history.append(
                {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
            )
            print(
                f"epoch {epoch:02d}/{epochs}  "
                f"train={train_loss:.4f}  val={val_loss:.4f}"
            )

            # rolling checkpoint every epoch (handy if the run gets killed)
            save_checkpoint(
                self.checkpoint_dir / "last.pt",
                self.model,
                self.vocab,
                self.optimizer,
                epoch,
                val_loss,
                self.meta,
            )

            if val_loss < best_val:
                best_val = val_loss
                bad_epochs = 0
                save_checkpoint(
                    best_path,
                    self.model,
                    self.vocab,
                    self.optimizer,
                    epoch,
                    val_loss,
                    self.meta,
                )
                print(f"  saved best -> {best_path}")
            else:
                bad_epochs += 1
                if bad_epochs >= self.patience:
                    print(f"early stop after {epoch} epochs (patience={self.patience})")
                    break

        return best_path
