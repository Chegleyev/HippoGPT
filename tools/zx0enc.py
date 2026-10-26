"""Упаковщик в формат ZX0 v2 под распаковщик dzx0_standard (Эйнар Сауказ, third_party/zx0).

Разбор жадный (не оптимальный, как у оригинального zx0): для экрана разница — десятки байт. Формат — как его
читает dzx0_standard:
  * биты идут старшим вперёд в отдельных байтах, которые вставляются в поток в момент, когда распаковщику
    понадобился бит, а прежний байт кончился;
  * число — чередующийся гамма-код Элиаса: для каждого бита ниже старшего «0, бит», в конце «1»;
  * блоки: литералы (длина, байты) -> бит 0: повтор прежнего смещения (длина) / бит 1: новое смещение;
    после совпадения бит 0 — литералы, 1 — новое смещение;
  * новое смещение: гамма-код (старшая часть + 1) с инвертированными битами данных, байт
    ((127 - младшие 7 бит (смещение - 1)) << 1) | первый управляющий бит длины, остаток кода (длина - 1);
  * конец — новое смещение с кодом 256.
"""
from __future__ import annotations


class _Out:
    def __init__(self):
        self.buf = bytearray()
        self.mask = 0
        self.at = 0

    def bit(self, b):
        if not self.mask:
            self.at = len(self.buf)
            self.buf.append(0)
            self.mask = 0x80
        if b:
            self.buf[self.at] |= self.mask
        self.mask >>= 1

    def elias(self, v, invert=False, first_to=None):
        """first_to — индекс байта, в младший бит которого идёт первый управляющий бит."""
        seq = []
        for i in range(v.bit_length() - 2, -1, -1):
            seq += [0, ((v >> i) & 1) ^ invert]
        seq.append(1)
        if first_to is not None:
            self.buf[first_to] |= seq[0]
            seq = seq[1:]
        for b in seq:
            self.bit(b)


def compress(data: bytes, max_offset: int = 32640, chain: int = 256) -> bytes:
    n = len(data)
    o = _Out()
    heads: dict[bytes, list[int]] = {}
    last, i, lit_start = 1, 0, 0
    after_match = None                      # None — в начале, False — после литералов, True — после совпадения

    def add_hash(p):
        if p + 2 < n:
            heads.setdefault(data[p:p + 3], []).append(p)

    def match_len(p, off):
        k = 0
        while p + k < n and data[p + k] == data[p + k - off]:
            k += 1
        return k

    def flush_literals(end):
        nonlocal after_match
        if end > lit_start:
            if after_match:
                o.bit(0)
            o.elias(end - lit_start)
            o.buf += data[lit_start:end]
            after_match = False

    while i < n:
        best_len, best_off = 0, 0
        if i > 0:
            if i > lit_start:                    # повтор смещения — только сразу после литералов
                rl = match_len(i, last) if i >= last else 0
                if rl >= 2:
                    best_len, best_off = rl, last
            for p in reversed(heads.get(data[i:i + 3], [])[-chain:]):
                off = i - p
                if off > max_offset:
                    break
                k = match_len(i, off)
                if k > best_len:
                    best_len, best_off = k, off
        if best_len >= 2 and i > 0:
            flush_literals(i)
            if after_match is False and best_off == last:
                o.bit(0)                     # повтор прежнего смещения
                o.elias(best_len)
            else:
                o.bit(1)
                o.elias((best_off - 1) // 128 + 1, invert=True)
                o.buf.append((127 - ((best_off - 1) & 127)) << 1)
                o.elias(best_len - 1, first_to=len(o.buf) - 1)
            last = best_off
            for p in range(i, i + best_len):
                add_hash(p)
            i += best_len
            lit_start = i
            after_match = True
        else:
            if after_match is None:
                after_match = False
            add_hash(i)
            i += 1
    flush_literals(n)
    o.bit(1)                                 # конец: новое смещение с кодом 256
    o.elias(256, invert=True)
    return bytes(o.buf)
