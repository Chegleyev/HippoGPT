"""Чекпойнт -> build/intmodel.pkl (целочисленная модель). Нужен torch: окружение обучения."""
import pickle
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import intref

if __name__ == "__main__":
    ck = torch.load(sys.argv[1], map_location="cpu")
    im = intref.export(ck)
    out = Path(sys.argv[2])
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(im, f)
    print("step", ck["step"], "val_bpc", round(ck["val_bpc"], 4), "emb_shift", im.emb_shift, "->", out)
