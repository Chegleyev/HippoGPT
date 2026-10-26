"""Сверка прямого прохода HippoGPT 128K с целочисленным эталоном: все логиты на каждом токене.

    python tools/test_forward128.py [число токенов]
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ref"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "train"))
import intref  # noqa: E402
import bpe  # noqa: E402
from zx128 import BUILD, ROOT, Machine128  # noqa: E402


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    with open(BUILD / "intmodel128.pkl", "rb") as f:
        im = pickle.load(f)
    meta = json.loads((ROOT / "data" / f"{im.data}.json").read_text(encoding="utf-8"))
    enc = bpe.encoder([tuple(x) for x in meta["merges"]], meta["pieces"])
    text = "Once upon a time, there was a little girl named Lily. She loved to play with her ball."
    toks = [0] + [t for p in bpe.PRE.findall(text) for t in enc(p)]
    if n > len(toks):                                     # длинная цепочка — из проверочного текста
        val = np.fromfile(ROOT / "data" / f"{im.data}_valid.u16", dtype=np.uint16)
        toks = [0] + val[1000:1000 + n - 1].astype(int).tolist()
    toks = toks[:n]
    print("токены:", "|".join(meta["pieces"][t] for t in toks[:24]), "..." if len(toks) > 24 else "")
    m = Machine128()
    S = m.syms
    m.call("init_sq")
    m.call("reset_model")
    r = intref.Runner(im, 1)
    fails, tot = 0, 0
    t0 = time.time()
    for pos, tok in enumerate(toks):
        r.trace = {}
        ref = r.step(np.array([tok]))[0]
        t = m.call("forward", hl=tok)
        tot += t
        got = np.array(m.peek_words(S["LOGITS"], im.vocab))
        ok = np.array_equal(got, ref)
        if not ok:
            fails += 1
            bad = np.nonzero(got != ref)[0]
            h = np.array(m.peek_words(S["HVEC"], im.d))
            print(f"  pos {pos}: {len(bad)} логитов расходятся, первые {bad[:6]} z80 {got[bad[:4]]} ref {ref[bad[:4]]};"
                  f" HVEC==ref h_mlp последнего слоя: {np.array_equal(h, r.trace[f'l{len(im.layers) - 1}.h_mlp'][0])}")
            if fails > 3:
                break
    dt = time.time() - t0
    print(("OK" if not fails else f"{fails} расхождений"),
          f"| {tot / len(toks) / 1e6:.2f} млн тактов на токен = {tot / len(toks) / 3.5e6:.2f} с на 3,5 МГц"
          f" | симулятор {dt / len(toks) * 1000:.0f} мс на токен")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
