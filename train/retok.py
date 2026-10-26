"""Перекодировать корпус готовым словарём BPE (без обучения слияний), параллельно.

    python train/retok.py [V=512] [процессов=8]   -> data/tok{V}_{valid,train}.u16

Куски режутся по концам сказок (байт 0), так что слово на стыке не рвётся.
"""
from __future__ import annotations

import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bpe  # noqa: E402

DATA = bpe.DATA
CH = 20_000_000
_enc = None
_src = None


def init(V, split):
    global _enc, _src
    meta = json.loads((DATA / f"tok{V}.json").read_text(encoding="utf-8"))
    _enc = bpe.encoder([tuple(m) for m in meta["merges"]], meta["pieces"])
    _src = np.memmap(DATA / f"{split}.u8", dtype=np.uint8, mode="r")


def work(span):
    lo, hi = span
    text = bpe.ids_to_text(np.asarray(_src[lo:hi]))
    parts = bpe.PRE.findall(text)
    assert sum(len(p) for p in parts) == len(text)
    return np.fromiter((t for p in parts for t in _enc(p)), dtype=np.uint16), hi - lo


def spans(src):
    out, lo = [], 0
    while lo < len(src):
        hi = min(lo + CH, len(src))
        if hi < len(src):
            nz = np.flatnonzero(np.asarray(src[hi:hi + 1_000_000]) == 0)
            hi = hi + int(nz[0]) + 1 if len(nz) else hi
        out.append((lo, hi))
        lo = hi
    return out


def main():
    V = int(sys.argv[1]) if len(sys.argv) > 1 else 512
    procs = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    for split in ("valid", "train"):
        src = np.memmap(DATA / f"{split}.u8", dtype=np.uint8, mode="r")
        tmp = DATA / f"tok{V}_{split}.u16.tmp"
        t0, chars, toks, eos = time.time(), 0, 0, 0
        with open(tmp, "wb") as f, Pool(procs, initializer=init, initargs=(V, split)) as pool:
            for ids, n in pool.imap(work, spans(src)):
                ids.tofile(f)
                chars += n
                toks += len(ids)
                eos += int((ids == 0).sum())
                print(f"{split}: {chars / 1e6:.0f}M символов, концов сказок {eos} ({time.time() - t0:.0f} с)", flush=True)
        os.replace(tmp, DATA / f"tok{V}_{split}.u16")
        print(f"{split}: {toks} токенов, {chars / toks:.3f} символа на токен, концов сказок {eos}", flush=True)


if __name__ == "__main__":
    main()
