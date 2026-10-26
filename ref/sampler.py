"""Целочисленная выборка токена и разбор затравки — ровно как на Z80.

Выборка:
  z_v   = acc_v * LM_v                    (LM — 8-битный масштаб строки выходного слоя)
  d_v   = zmax - z_v
  u_v   = 255, если d_v > 65535, иначе min(255, (d_v * TQ) >> 16)    (TQ — температура, 16 бит)
  p_v   = EXP127[u_v]                     (127*exp(-u/16): сумма по 512 токенам < 65536)
  r     = xorshift16(...)                 (генератор Меткалфа 7/9/8, как в коде Z80)
  порог = (r * сумма) >> 16; токен — первый, на котором накопленная сумма > порога.

Склейка (token healing): последний токен затравки в модель не подаётся; перед первой выборкой токенам,
чья строка не начинается с его строки, ставится метка acc = -32768 (обычные |acc| < 2500) -> p = 0.

Разбор затравки — слияниями BPE, как при обучении: куски bpe.PRE, в куске — символы, затем один проход по
слияниям в порядке номеров, каждое — по всем парам слева направо. Это то же, что «сначала пара с меньшим
номером» в bpe.encoder: слияние с новым токеном 64+r всегда имеет номер больше r.
"""
from __future__ import annotations

import math
import re

import numpy as np

EXP127 = np.array([int(math.floor(127 * math.exp(-u / 16) + 0.5)) for u in range(256)], dtype=np.int64)
PRE = re.compile(r" ?[A-Za-z]+| ?[^A-Za-z \n\0]+|\n|\0| ")


def logit_mults(logit_row: np.ndarray):
    """-> (LM: uint8 на строку, unit_z: натов на единицу z)."""
    unit_z = float(logit_row.max()) / 255
    lm = np.clip(np.floor(logit_row / unit_z + 0.5), 1, 255).astype(np.int64)
    return lm, unit_z


def temp_q(unit_z: float, temp: float) -> int:
    tq = int(round(unit_z * 16 / temp * 65536))
    assert 0 < tq < 65536, tq
    return tq


def xorshift16(x: int) -> int:
    """Тот же порядок операций, что у Z80-кода (Джон Меткалф, сдвиги 7, 9, 8)."""
    x ^= (x << 7) & 0xFFFF
    x ^= x >> 9
    x ^= (x << 8) & 0xFFFF
    return x & 0xFFFF


def sample(acc: np.ndarray, lm: np.ndarray, tq: int, rng: int):
    """-> (токен, новое состояние генератора)."""
    z = acc.astype(np.int64) * lm
    d = z.max() - z
    u = np.where(d > 65535, 255, np.minimum(255, (d * tq) >> 16))
    u = np.where(acc == HEAL_MASK, 255, u)          # метка склейки: p = EXP127[255] = 0
    p = EXP127[u]
    total = int(p.sum())
    assert total < 65536
    rng = xorshift16(rng)
    thr = (rng * total) >> 16
    cum = np.cumsum(p)
    return int(np.argmax(cum > thr)), rng


HEAL_MASK = -32768
assert EXP127[255] == 0


def heal_mask(acc: np.ndarray, pieces: list[str], need: int) -> np.ndarray:
    """Первая выборка после склейки: только токены, чья строка начинается со строки need."""
    acc = acc.copy()
    ok = np.array([pc.startswith(pieces[need]) for pc in pieces])
    acc[~ok] = HEAL_MASK
    return acc


VOCAB = ["\0", "\n", " ", "!", '"', "'", ",", "-", ".", ":", ";", "?"] + \
        [chr(c) for c in range(ord("A"), ord("Z") + 1)] + [chr(c) for c in range(ord("a"), ord("z") + 1)]


def tokenize_bpe(text: str, merges: list) -> list[int]:
    """Как на Z80: по кускам, слияния по порядку, каждое — все пары слева направо."""
    out = []
    for chunk in PRE.findall(text):
        seq = [VOCAB.index(c) for c in chunk]
        for r, (a, b) in enumerate(merges):
            i = 0
            while i < len(seq) - 1:
                if seq[i] == a and seq[i + 1] == b:
                    seq[i] = len(VOCAB) + r
                    del seq[i + 1]
                i += 1
        out += seq
    return out


def tokenize_greedy(text: str, pieces: list[str]) -> list[int]:
    by_len = sorted(range(len(pieces)), key=lambda i: -len(pieces[i]))
    out = []
    for chunk in PRE.findall(text):
        i = 0
        while i < len(chunk):
            for t in by_len:
                p = pieces[t]
                if t and p and chunk.startswith(p, i):
                    out.append(t)
                    i += len(p)
                    break
            else:
                raise ValueError(f"нет токена для {chunk[i:]!r}")
    return out
