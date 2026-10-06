# LSTM Text Generation

Word-level LSTM that learns next-token prediction on Shakespeare's complete works
(Project Gutenberg #100), then samples new text from a seed phrase.

Built as a small Python package (`textgen/`) so data prep, the model, training,
and generation stay in separate modules instead of one notebook dump.

## Layout

```
ga_with_lstm/
├── main.py                 # CLI entrypoint
├── textgen/
│   ├── config.py           # env-backed settings
│   ├── data/               # download, clean, vocab, Dataset
│   ├── model/              # LSTM architecture
│   ├── training/           # loop, early stop, checkpoints
│   ├── generation/         # seed → sampled text
│   └── utils/
├── data/                   # corpus lands here after download
├── checkpoints/            # best.pt / last.pt
├── samples/                # generated_samples.txt
├── .env.example
└── requirements.txt
```

## Setup

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
copy .env.example .env      # Windows
# cp .env.example .env      # macOS / Linux
```

Tweaks go in `.env` (epochs, sequence length, hidden size, etc.). CLI flags
override those values when you pass them.

## Dataset

Default corpus: [Shakespeare's Complete Works](https://www.gutenberg.org/files/100/100-0.txt)
(Project Gutenberg, public domain).

```bash
python main.py download
```

Or drop any large `.txt` into `data/` and point `DATA_PATH` / `--data` at it.

Preprocessing (in `textgen/data/`):

1. Lowercase + strip punctuation
2. Whitespace tokenize into words
3. Build a vocab (words seen fewer than twice → `<unk>`)
4. Sliding windows of length `SEQ_LENGTH` → predict the next word

## Train

```bash
python main.py train
```

Useful knobs:

```bash
python main.py train --epochs 20 --num-layers 3 --seq-length 60 --hidden-dim 512
```

What the trainer does:

- 90/10 train/val split
- Adam + cross-entropy
- gradient clipping at 1.0
- writes `checkpoints/last.pt` every epoch
- keeps `checkpoints/best.pt` on lowest val loss
- early stopping when val loss stalls (`EARLY_STOP_PATIENCE`)

Full pipeline in one shot:

```bash
python main.py run --epochs 10
```

## Generate

```bash
python main.py generate --seed "to be or not to be"
python main.py generate --seed "friends romans countrymen" --temperature 0.7 --length 150
python main.py generate --samples
```

`--samples` runs a few fixed seeds and writes `samples/generated_samples.txt`.

Temperature: lower → safer / more repetitive; higher → wilder.

## Architecture

```
Embedding(vocab, embed_dim)
  → Dropout
  → LSTM(embed_dim → hidden_dim, num_layers, dropout between layers)
  → Dropout
  → Linear(hidden_dim → vocab)   # logits for next word
```

Default: `embed=128`, `hidden=256`, `layers=2`, `seq=40`, `dropout=0.3`.

### Bonus — what I tried

| Change | Observation |
|--------|-------------|
| 1 LSTM layer | Trains faster, text stays more generic / less rhythmic |
| 3 layers + dropout 0.4 | Slightly better local phrasing, slower, overfits sooner on short runs |
| `seq_length=20` | Snappier training, weaker long-range sense |
| `seq_length=60` | Heavier batches, better continuity when it converges |
| `temperature=0.5` vs `1.2` | 0.5 repeats safe Shakespeare-ish words; 1.2 invents more nonsense |

For a quick architecture A/B on your machine:

```bash
python main.py train --num-layers 1 --epochs 8
python main.py train --num-layers 3 --hidden-dim 384 --epochs 8
```

Compare `samples/` after generating from each `best.pt` (rename checkpoints between runs so they don't overwrite).

## Example output

From a short smoke run (`--max-tokens 30000 --epochs 2`). Full-corpus training
will sound more coherent; this is just to show the pipeline works:

```
=== seed: 'to be or not to be' ===
to be or not to be even that earth truth to the unfathered and the lord
of of this ... honour in thee said forth ... sweet helena was you this ...

=== seed: 'friends romans countrymen' ===
friends than well of thee alone most shade vile with me must in a just
steal king clown have upon let you ...
```

Re-generate anytime with `python main.py generate --samples`.

## Requirements

- Python 3.10+
- PyTorch 2.x (CUDA optional; CPU works, just slower)

## Notes

- First download needs network access to gutenberg.org
- On CPU, start with `--epochs 5 --batch-size 32` to sanity-check the pipeline
- Checkpoints store the vocab + model hyperparams, so generation doesn't need the raw corpus
