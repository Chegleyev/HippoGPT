"""Сэмплы из чекпойнта: python train/sample.py <ckpt> [n] [temp] [prompt]"""
import sys

import torch

from model import TinyGPT, cfg_from
from prep_data import VOCAB


def codec(ck):
    """(кодировщик затравки, строки токенов)."""
    if not ck.get("data"):
        return (lambda s: [VOCAB.index(c) for c in s]), VOCAB
    import json
    from pathlib import Path
    import bpe
    meta = json.loads((Path(__file__).resolve().parent.parent / "data" / f"{ck['data']}.json").read_text(encoding="utf-8"))
    enc = bpe.encoder([tuple(m) for m in meta["merges"]], meta["pieces"])
    return (lambda s: [t for p in bpe.PRE.findall(s) for t in enc(p)]), meta["pieces"]


@torch.no_grad()
def generate(m, prompt, n_chars, temp, gen, ctx, encode, pieces):
    ids = [0] + encode(prompt)
    out = prompt
    while len(out) < len(prompt) + n_chars:
        x = torch.tensor([ids[-ctx:]])
        logits = m(x)[0, -1] / temp
        p = torch.softmax(logits, -1)
        t = int(torch.multinomial(p, 1, generator=gen))
        if t == 0:
            break
        ids.append(t)
        out += pieces[t]
    return out


if __name__ == "__main__":
    ck = torch.load(sys.argv[1], map_location="cpu")
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    temp = float(sys.argv[3]) if len(sys.argv) > 3 else 0.8
    prompt = sys.argv[4] if len(sys.argv) > 4 else "Once upon a time"
    cfg = cfg_from(ck["cfg"])
    m = TinyGPT(cfg)
    m.load_state_dict(ck["model"])
    m.eval()
    gen = torch.Generator().manual_seed(7)
    encode, pieces = codec(ck)
    for i in range(n):
        print(f"--- {i + 1}")
        print(generate(m, prompt, 500, temp, gen, cfg.ctx, encode, pieces))
