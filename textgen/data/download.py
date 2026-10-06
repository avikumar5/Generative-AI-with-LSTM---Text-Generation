"""Fetch the training corpus if it isn't on disk yet."""

from __future__ import annotations

from pathlib import Path

import requests

from textgen.config import DATA_PATH, GUTENBERG_URL


def ensure_dataset(dest: Path | None = None, url: str = GUTENBERG_URL) -> Path:
    """
    Download Shakespeare's complete works (Gutenberg #100) unless `dest` exists.

    Strips the Gutenberg header/footer so we keep the literary text only.
    """
    dest = dest or DATA_PATH
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and dest.stat().st_size > 10_000:
        return dest

    print(f"Downloading corpus -> {dest}")
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    # Gutenberg ships UTF-8 with a BOM sometimes
    raw = resp.content.decode("utf-8-sig", errors="replace")

    start = raw.find("*** START OF")
    end = raw.find("*** END OF")
    if start != -1 and end != -1:
        # jump past the START marker line
        start = raw.find("\n", start) + 1
        body = raw[start:end].strip()
    else:
        body = raw

    dest.write_text(body, encoding="utf-8")
    print(f"Saved {len(body):,} characters")
    return dest
