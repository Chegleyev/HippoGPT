"""Истории про Спекки -> токены tok512 для финального дообучения.

    python train/prep_speccy.py      -> data/speccy512_{train,valid}.u16

Источник: data/speccy/speccy_stories.txt — 1613 сказок про ZX Spectrum, написанных gemma4:12b (Ollama) и
Claude Haiku по описанию с фактами о Спектруме в стиле TinyStories.
Каждая история проходит тот же фильтр clean(), повторы по началу (60 символов) выбрасываются.
Формат как у корпуса: история, затем 0 (конец истории). 100 историй — отложенная проверка.
"""
from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bpe  # noqa: E402
from prep_data import FOLD, VOCAB, encode_text, make_table  # noqa: E402

ALLOWED = set(VOCAB[1:])


def clean(text: str) -> str | None:
    for a, b in FOLD.items():
        text = text.replace(a, b)
    text = text.replace("\r", "").strip()
    text = re.sub(r"^(Title|Story)[^\n]*\n+", "", text)
    text = re.sub(r"\*+", "", text)
    text = re.sub(r"\n{2,}", "\n", text).strip()
    if re.search(r"\d", text) or not (450 <= len(text) <= 1200):
        return None
    if any(ch not in ALLOWED for ch in text):
        return None
    if not re.search(r"Speccy|Spectrum|Hippo", text):
        return None
    return text


ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "speccy"
N_VALID = 100


def stories():
    seen, out = set(), []
    for s in (SRC / "speccy_stories.txt").read_text(encoding="utf-8").split("<|endoftext|>"):
        if not s.strip():
            continue
        c = clean(s)
        if c is None or c[:60] in seen:
            continue
        seen.add(c[:60])
        out.append(c)
    return out


def main():
    all_ = stories()
    print("историй:", len(all_))
    random.Random(2026).shuffle(all_)
    meta = json.loads((ROOT / "data" / "tok512.json").read_text(encoding="utf-8"))
    enc = bpe.encoder([tuple(m) for m in meta["merges"]], meta["pieces"])
    lut = make_table()
    for split, part in [("valid", all_[:N_VALID]), ("train", all_[N_VALID:])]:
        text = "".join(s + "\n<|endoftext|>\n" for s in part)
        chars = bpe.ids_to_text(encode_text(text, lut))
        pieces = bpe.PRE.findall(chars)
        assert "".join(pieces) == chars
        ids = np.fromiter((t for p in pieces for t in enc(p)), dtype=np.uint16)
        ids.tofile(ROOT / "data" / f"speccy512_{split}.u16")
        print(f"{split}: {len(part)} историй, {len(chars)} символов, {len(ids)} токенов")


if __name__ == "__main__":
    main()
