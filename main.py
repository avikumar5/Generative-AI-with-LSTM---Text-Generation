#!/usr/bin/env python3
"""
CLI for the LSTM text-generation pipeline.

  python main.py download
  python main.py train
  python main.py generate --seed "to be or not to be"
  python main.py run          # download → train → sample
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# allow `python main.py` from the repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent))

from textgen import config
from textgen.data import build_dataloaders, ensure_dataset
from textgen.generation import generate_samples, generate_text
from textgen.model import LSTMTextGenerator
from textgen.training import Trainer, load_checkpoint
from textgen.utils import get_device


def cmd_download(_: argparse.Namespace) -> None:
    path = ensure_dataset()
    print(f"ready: {path}")


def cmd_train(args: argparse.Namespace) -> None:
    data_path = ensure_dataset(Path(args.data) if args.data else None)
    train_loader, val_loader, vocab = build_dataloaders(
        data_path,
        seq_len=args.seq_length,
        batch_size=args.batch_size,
        val_split=config.VAL_SPLIT,
        max_tokens=args.max_tokens,
    )

    model = LSTMTextGenerator(
        vocab_size=len(vocab),
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )

    meta = {
        "embed_dim": args.embed_dim,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "seq_length": args.seq_length,
    }

    trainer = Trainer(
        model,
        train_loader,
        val_loader,
        vocab,
        lr=args.lr,
        patience=args.patience,
        meta=meta,
    )
    best = trainer.fit(epochs=args.epochs)
    print(f"done. best checkpoint: {best}")


def cmd_generate(args: argparse.Namespace) -> None:
    device = get_device()
    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        raise SystemExit(f"checkpoint not found: {ckpt}")

    model, vocab, state = load_checkpoint(ckpt, device=device)
    seq_len = state.get("meta", {}).get("seq_length", config.SEQ_LENGTH)
    model._seq_len = seq_len  # used by the rolling window in generate_text

    if args.samples:
        report = generate_samples(
            model,
            vocab,
            length=args.length,
            temperature=args.temperature,
        )
        print(report)
    else:
        text = generate_text(
            model,
            vocab,
            seed=args.seed,
            length=args.length,
            temperature=args.temperature,
            device=device,
        )
        print(text)


def cmd_run(args: argparse.Namespace) -> None:
    cmd_download(args)
    cmd_train(args)
    # after training, sample from best.pt
    args.checkpoint = str(config.CHECKPOINT_DIR / "best.pt")
    args.samples = True
    cmd_generate(args)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="LSTM word-level text generation (Shakespeare)"
    )
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("download", help="fetch the Gutenberg Shakespeare corpus")
    d.set_defaults(func=cmd_download)

    t = sub.add_parser("train", help="preprocess + train LSTM")
    t.add_argument("--data", default=None, help="path to .txt corpus")
    t.add_argument("--epochs", type=int, default=config.EPOCHS)
    t.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    t.add_argument("--seq-length", type=int, default=config.SEQ_LENGTH)
    t.add_argument("--embed-dim", type=int, default=config.EMBED_DIM)
    t.add_argument("--hidden-dim", type=int, default=config.HIDDEN_DIM)
    t.add_argument("--num-layers", type=int, default=config.NUM_LAYERS)
    t.add_argument("--dropout", type=float, default=config.DROPOUT)
    t.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    t.add_argument("--patience", type=int, default=config.EARLY_STOP_PATIENCE)
    t.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="cap corpus size (handy for smoke tests)",
    )
    t.set_defaults(func=cmd_train)

    g = sub.add_parser("generate", help="sample text from a checkpoint")
    g.add_argument(
        "--checkpoint",
        default=str(config.CHECKPOINT_DIR / "best.pt"),
    )
    g.add_argument("--seed", default=config.SEED_TEXT)
    g.add_argument("--length", type=int, default=config.GEN_LENGTH)
    g.add_argument("--temperature", type=float, default=config.TEMPERATURE)
    g.add_argument(
        "--samples",
        action="store_true",
        help="write several seed demos to samples/generated_samples.txt",
    )
    g.set_defaults(func=cmd_generate)

    r = sub.add_parser("run", help="download → train → generate samples")
    r.add_argument("--data", default=None)
    r.add_argument("--epochs", type=int, default=config.EPOCHS)
    r.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    r.add_argument("--seq-length", type=int, default=config.SEQ_LENGTH)
    r.add_argument("--embed-dim", type=int, default=config.EMBED_DIM)
    r.add_argument("--hidden-dim", type=int, default=config.HIDDEN_DIM)
    r.add_argument("--num-layers", type=int, default=config.NUM_LAYERS)
    r.add_argument("--dropout", type=float, default=config.DROPOUT)
    r.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    r.add_argument("--patience", type=int, default=config.EARLY_STOP_PATIENCE)
    r.add_argument("--max-tokens", type=int, default=None)
    r.add_argument("--length", type=int, default=config.GEN_LENGTH)
    r.add_argument("--temperature", type=float, default=config.TEMPERATURE)
    r.set_defaults(func=cmd_run)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
