"""Sample text from a trained LSTM given a seed phrase."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from textgen.config import GEN_LENGTH, SAMPLE_DIR, SEED_TEXT, TEMPERATURE
from textgen.data.preprocess import Vocab, tokenize
from textgen.model.lstm import LSTMTextGenerator


def _temperature_sample(
    logits: torch.Tensor,
    temperature: float,
    ban: set[int] | None = None,
) -> int:
    if ban:
        logits = logits.clone()
        for idx in ban:
            logits[idx] = float("-inf")
    if temperature <= 0:
        return int(logits.argmax(dim=-1).item())
    probs = F.softmax(logits / temperature, dim=-1)
    return int(torch.multinomial(probs, 1).item())


@torch.no_grad()
def generate_text(
    model: LSTMTextGenerator,
    vocab: Vocab,
    seed: str = SEED_TEXT,
    length: int = GEN_LENGTH,
    temperature: float = TEMPERATURE,
    device: torch.device | None = None,
) -> str:
    """
    Feed `seed`, then keep predicting the next word `length` times.
    Temperature softens ( >1 ) or sharpens ( <1 ) the distribution.
    """
    device = device or next(model.parameters()).device
    model.eval()

    tokens = tokenize(seed)
    if not tokens:
        tokens = ["the"]

    ids = vocab.encode(tokens)
    seq_len = getattr(model, "_seq_len", None)
    generated = list(ids)
    ban = {vocab.stoi[Vocab.PAD], vocab.stoi[Vocab.UNK]}

    for _ in range(length):
        ctx = generated[-(seq_len or len(generated)) :]
        x = torch.tensor([ctx], dtype=torch.long, device=device)
        logits, _ = model(x)
        nxt = _temperature_sample(logits[0], temperature, ban=ban)
        generated.append(nxt)

    # drop special tokens if they somehow sneak in via the seed path
    words = [
        vocab.itos[i]
        for i in generated
        if i < len(vocab.itos) and vocab.itos[i] not in (Vocab.PAD, Vocab.UNK)
    ]
    return " ".join(words)


def generate_samples(
    model: LSTMTextGenerator,
    vocab: Vocab,
    seeds: list[str] | None = None,
    length: int = GEN_LENGTH,
    temperature: float = TEMPERATURE,
    out_path: Path | None = None,
) -> str:
    seeds = seeds or [
        SEED_TEXT,
        "what light through yonder",
        "friends romans countrymen",
        "all the worlds a stage",
    ]
    blocks = []
    for seed in seeds:
        text = generate_text(model, vocab, seed=seed, length=length, temperature=temperature)
        blocks.append(f"=== seed: {seed!r} ===\n{text}\n")

    report = "\n".join(blocks)
    if out_path is None:
        SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
        out_path = SAMPLE_DIR / "generated_samples.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    return report
