"""Сравнить bpc: QAT-модель (torch) и целочисленный эталон на одних окнах valid.

    python ref/check.py <ckpt> [окон] [корпус: tok512 | speccy512]
"""
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import intref

intref.sys_path_model()
from model import Cfg, TinyGPT, cfg_from

ROOT = Path(__file__).resolve().parent.parent


def qat_bpc(ckpt, data, n_windows, T, piece_len=None):
    m = TinyGPT(cfg_from(ckpt["cfg"]))
    m.load_state_dict(ckpt["model"])
    m.eval()
    starts = np.arange(n_windows) * T
    chunk = torch.from_numpy(data[starts[:, None] + np.arange(T + 1)[None]].astype(np.int64))
    p0 = torch.from_numpy(np.where(starts > 0, data[np.maximum(starts - 1, 0)], 0).astype(np.int64))[:, None]
    with torch.no_grad():
        _, loss = m(chunk[:, :-1], chunk[:, 1:], p0)
    n = chunk[:, 1:].numel()
    chars = int(piece_len[chunk[:, 1:].numpy()].sum()) if piece_len is not None else n
    return float(loss) * n / chars / math.log(2)


if __name__ == "__main__":
    ck = torch.load(sys.argv[1], map_location="cpu")
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 512
    piece_len = None
    if ck.get("data"):
        import json
        meta = json.loads((ROOT / "data" / f"{ck['data']}.json").read_text(encoding="utf-8"))
        piece_len = np.array([len(p) for p in meta["pieces"]])
        vname = sys.argv[3] if len(sys.argv) > 3 else ck["data"]    # другой корпус, напр. speccy512
        data = np.fromfile(ROOT / "data" / f"{vname}_valid.u16", dtype=np.uint16)
    else:
        data = np.fromfile(ROOT / "data" / "valid.u8", dtype=np.uint8)
    T = ck["cfg"]["ctx"]
    print("step", ck["step"], "train-eval bpc", ck["val_bpc"])
    print(f"QAT  bpc ({n} окон): {qat_bpc(ck, data, n, T, piece_len):.4f}")
    im = intref.export(ck)
    for i, ly in enumerate(im.layers):
        print(f"  слой {i}: сдвиги qkv {ly.sqkv} o {ly.so} w1 {ly.s1} w2 {ly.s2}; bias {ly.bias.min()}..{ly.bias.max()}")
    t0 = time.time()
    print(f"INT  bpc ({n} окон): {intref.eval_bpc(im, data, n, T, piece_len=piece_len):.4f}  ({time.time() - t0:.0f} с)")
    print("макс. |аккумулятор| по матрицам:", intref.ACC_MAX)
    if im.emb_mul is not None:
        print("множители вложений:", im.emb_mul.min(), "..", im.emb_mul.max())
    for nm in ("prev", "big"):
        if getattr(im, nm) is not None:
            mu = getattr(im, nm + "_mul")
            print(f"множители {nm}:", mu.min(), "..", mu.max())
