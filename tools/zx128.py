"""Стенд Z80 для 128K: восемь банков, переключение через порт 7FFD (трассировщик SkoolKit), вызов подпрограмм."""
from __future__ import annotations

import re
from pathlib import Path

from skoolkit import CSimulator
from skoolkit.pagingtracer import Memory, PagingTracer
from skoolkit.simutils import A, B, C, D, E, H, L, IXh, IXl, IYh, IYl, PC, SP, T, from_memory

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "build"
STOP = 0x5D00          # ловушка возврата: низ стека, туда программа не прыгает


def load_syms(path=BUILD / "hippo.sym"):
    syms = {}
    for line in Path(path).read_text().splitlines():
        m = re.match(r"(\S+):\s+EQU\s+0x([0-9A-Fa-f]+)", line)
        if m:
            syms[m.group(1)] = int(m.group(2), 16)
    return syms


class Machine128(PagingTracer):
    def __init__(self, syms=None, screen: bytes | None = None):
        banks = []
        for n in range(8):
            p = BUILD / f"bank{n}.bin"
            banks.append(list(p.read_bytes()) if p.exists() else [0] * 16384)
        if screen is not None:
            banks[5][:len(screen)] = list(screen)
        mem = Memory(banks, 0x10, "128K")
        self.simulator = from_memory(CSimulator, mem, {"PC": 0, "SP": 0x5E00}, {"iff": 0, "im": 1}, {})
        self.out7ffd = 0x10
        self.outfffd = 0
        self.ay = [0] * 16
        self.outfe = 0
        self.border = 0
        self.simulator.set_tracer(self)
        self.mem = self.simulator.memory
        self.reg = self.simulator.registers
        self.syms = syms or load_syms()
        self.keyboard = [0] * 8

    # --- память ---
    def bank(self, n):
        """Подключить банк n на C000 (для чтения из стенда)."""
        v = 0x10 | n
        self.simulator.memory.out7ffd(v)
        self.out7ffd = v

    def peek(self, addr, n) -> bytes:
        return bytes(self.mem[addr + i] for i in range(n))

    def poke(self, addr, data):
        for i, b in enumerate(data):
            self.mem[addr + i] = b & 0xFF

    def peek_words(self, addr, n, signed=True):
        out = []
        for i in range(n):
            v = self.mem[addr + 2 * i] | (self.mem[addr + 2 * i + 1] << 8)
            out.append(v - 0x10000 if signed and v >= 0x8000 else v)
        return out

    def poke_words(self, addr, values):
        for i, v in enumerate(values):
            v &= 0xFFFF
            self.mem[addr + 2 * i] = v & 0xFF
            self.mem[addr + 2 * i + 1] = v >> 8

    def peek_s8(self, addr, n):
        return [b - 256 if b >= 128 else b for b in self.peek(addr, n)]

    # --- вызов ---
    def call(self, routine, sp=0x5E00, max_t=10 ** 10, **regs) -> int:
        if isinstance(routine, str):
            routine = self.syms[routine]
        r = self.reg
        sp -= 2
        self.mem[sp] = STOP & 0xFF
        self.mem[sp + 1] = STOP >> 8
        r[SP] = sp
        pairs = {"bc": (B, C), "de": (D, E), "hl": (H, L), "ix": (IXh, IXl), "iy": (IYh, IYl)}
        for k, v in regs.items():
            if k in pairs:
                hi, lo = pairs[k]
                r[hi], r[lo] = (v >> 8) & 0xFF, v & 0xFF
            else:
                r[{"a": A, "b": B, "c": C, "d": D, "e": E, "h": H, "l": L}[k]] = v & 0xFF
        t0 = r[T]
        self.simulator.trace(routine, STOP, 0, t0 + max_t, False, None, None, self.keyboard, None, None)
        assert r[PC] == STOP, f"не вернулось: PC={r[PC]:#06x}"
        return r[T] - t0
