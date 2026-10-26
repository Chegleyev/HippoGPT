"""TinyStories -> поток символов словаря из 64 знаков (uint8).

Словарь рассчитан на шрифт ПЗУ Spectrum: только ASCII. Код 0 — конец истории.
Типографские кавычки и тире сводятся к ASCII, цифры и редкие знаки выбрасываются.
"""
import sys
from pathlib import Path

import numpy as np

VOCAB = ["\0", "\n", " ", "!", '"', "'", ",", "-", ".", ":", ";", "?"] + \
        [chr(c) for c in range(ord("A"), ord("Z") + 1)] + \
        [chr(c) for c in range(ord("a"), ord("z") + 1)]
assert len(VOCAB) == 64
EOS = 0

FOLD = {"“": '"', "”": '"', "‘": "'", "’": "'", "–": "-", "—": "-",
        "…": "."}


def make_table() -> np.ndarray:
    lut = np.full(0x110000 if False else 0x2100, 255, dtype=np.uint8)
    for i, ch in enumerate(VOCAB):
        lut[ord(ch)] = i
    for src, dst in FOLD.items():
        lut[ord(src)] = VOCAB.index(dst)
    return lut


def encode_text(text: str, lut: np.ndarray) -> np.ndarray:
    text = text.replace("\n<|endoftext|>\n", "\0").replace("<|endoftext|>", "\0")
    cps = np.frombuffer(text.encode("utf-32-le"), dtype=np.uint32)
    cps = np.where(cps < len(lut), cps, 0x20FF)          # всё за пределами таблицы -> выброс
    ids = lut[cps]
    ids = ids[ids != 255]
    # пустые строки подряд и пробелы по краям строк в корпусе редки — не трогаем
    return ids


def convert(src: Path, dst: Path, chunk=1 << 26):
    lut = make_table()
    total = 0
    with open(src, "rb") as f, open(dst, "wb") as out:
        tail = b""
        while True:
            buf = f.read(chunk)
            if not buf:
                break
            buf = tail + buf
            cut = buf.rfind(b"<|endoftext|>")
            if cut < 0 or len(buf) < chunk // 2:
                cut = len(buf)
            tail = buf[cut:]
            ids = encode_text(buf[:cut].decode("utf-8", errors="ignore"), lut)
            ids.tofile(out)
            total += len(ids)
            print(f"{dst.name}: {total/1e6:.0f}M", flush=True)
        if tail:
            ids = encode_text(tail.decode("utf-8", errors="ignore"), lut)
            ids.tofile(out)
            total += len(ids)
    print(dst, total)


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent / "data"
    convert(root / "TinyStoriesV2-GPT4-valid.txt", root / "valid.u8")
    if "--valid-only" not in sys.argv:
        convert(root / "TinyStoriesV2-GPT4-train.txt", root / "train.u8")
