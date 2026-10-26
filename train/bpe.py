"""BPE-токенизатор поверх 64-символьного алфавита и перекодировка корпуса.

    python train/bpe.py 256 512        -> data/tok{V}.json, data/tok{V}_{train,valid}.u16

Базовые токены 0..63 — символы VOCAB (0 — конец истории), слияния добавляют 64..V-1.
Предразбиение как у GPT-2: слово вместе с пробелом перед ним, пунктуация отдельно; слияния не
переходят границы кусков — так токены остаются кусками слов, а Z80 при выводе просто печатает их
строки.
"""
from __future__ import annotations

import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prep_data import VOCAB  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PRE = re.compile(r" ?[A-Za-z]+| ?[^A-Za-z \n\0]+|\n|\0| ")
# байт словаря -> ASCII; не через np.array(VOCAB): numpy-строки теряют "\0", и пропадали концы сказок
LUT = bytes(ord(VOCAB[i]) if i < len(VOCAB) else 0 for i in range(256))


def ids_to_text(a: np.ndarray) -> str:
    return np.asarray(a, dtype=np.uint8).tobytes().translate(LUT).decode("ascii")


def train_merges(word_counts: Counter, vocab_size: int):
    """Классический BPE по словарю частот кусков. -> список слияний (a, b) в порядке рангов."""
    idx = {ch: i for i, ch in enumerate(VOCAB)}
    words = [(tuple(idx[c] for c in w), n) for w, n in word_counts.items()]
    merges = []
    pieces = [VOCAB[i] for i in range(len(VOCAB))]
    while len(pieces) < vocab_size:
        pairs = Counter()
        for w, n in words:
            for a, b in zip(w, w[1:]):
                pairs[(a, b)] += n
        if not pairs:
            break
        (a, b), _ = pairs.most_common(1)[0]
        new = len(pieces)
        pieces.append(pieces[a] + pieces[b])
        merges.append((a, b))
        nw = []
        for w, n in words:
            if len(w) > 1:
                out, i = [], 0
                while i < len(w):
                    if i + 1 < len(w) and w[i] == a and w[i + 1] == b:
                        out.append(new)
                        i += 2
                    else:
                        out.append(w[i])
                        i += 1
                w = tuple(out)
            nw.append((w, n))
        words = nw
    return merges, pieces


def encoder(merges, pieces):
    rank = {m: r for r, m in enumerate(merges)}
    idx = {ch: i for i, ch in enumerate(VOCAB)}
    cache = {}

    def enc(word: str):
        t = cache.get(word)
        if t is not None:
            return t
        w = [idx[c] for c in word]
        while len(w) > 1:
            best = min(((rank.get((a, b), 1 << 30), i) for i, (a, b) in enumerate(zip(w, w[1:]))))
            if best[0] == 1 << 30:
                break
            i = best[1]
            w = w[:i] + [len(VOCAB) + best[0]] + w[i + 2:]
        cache[word] = w
        return w

    return enc


def main():
    sizes = [int(v) for v in sys.argv[1:]] or [256, 512]
    raw = np.memmap(DATA / "train.u8", dtype=np.uint8, mode="r")
    t0 = time.time()
    sample = ids_to_text(np.asarray(raw[:200_000_000]))
    wc = Counter(PRE.findall(sample))
    print(f"кусков: {len(wc)} уникальных, {sum(wc.values())} всего ({time.time() - t0:.0f} с)", flush=True)
    encs = {}
    for V in sizes:
        merges, pieces = train_merges(wc, V)
        (DATA / f"tok{V}.json").write_text(json.dumps({"pieces": pieces, "merges": merges}), encoding="utf-8")
        encs[V] = encoder(merges, pieces)
        print(f"V={V}: последние токены {pieces[-12:]} ({time.time() - t0:.0f} с)", flush=True)
    for split in ("valid", "train"):
        src = np.memmap(DATA / f"{split}.u8", dtype=np.uint8, mode="r")
        outs = {V: open(DATA / f"tok{V}_{split}.u16", "wb") for V in sizes}
        n_chars, n_tok = 0, {V: 0 for V in sizes}
        CH = 50_000_000
        for off in range(0, len(src), CH):
            text = ids_to_text(np.asarray(src[off:off + CH]))
            parts = PRE.findall(text)
            assert sum(len(p) for p in parts) == len(text)
            n_chars += len(text)
            for V in sizes:
                enc = encs[V]
                ids = np.fromiter((t for p in parts for t in enc(p)), dtype=np.uint16)
                ids.tofile(outs[V])
                n_tok[V] += len(ids)
            print(f"{split}: {n_chars / 1e6:.0f}M символов ({time.time() - t0:.0f} с)", flush=True)
        for V in sizes:
            outs[V].close()
            meta_p = DATA / f"tok{V}.json"
            meta = json.loads(meta_p.read_text(encoding="utf-8"))
            meta[f"chars_per_token_{split}"] = n_chars / n_tok[V]
            meta_p.write_text(json.dumps(meta), encoding="utf-8")
            print(f"V={V} {split}: {n_tok[V]} токенов, {n_chars / n_tok[V]:.3f} символа на токен", flush=True)


if __name__ == "__main__":
    main()
