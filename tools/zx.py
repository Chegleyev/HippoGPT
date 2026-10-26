"""Стенд Z80: сборка sjasmplus, загрузка в симулятор SkoolKit (C-ядро), вызов подпрограмм.

Вызов: на стек кладётся адрес-ловушка STOP, PC = подпрограмма, симулятор крутится до
возврата в STOP. Прерывания запрещены — программа на Спектруме тоже работает с DI.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

from skoolkit import CSimulator
from skoolkit.simutils import (A, B, C, D, E, H, L, IXh, IXl, IYh, IYl, PC, SP, T,
                               from_memory, read_bin_file, ROM48)

ROOT = Path(__file__).resolve().parent.parent
# ассемблер: переменная SJASMPLUS, иначе sjasmplus из PATH
SJASM = os.environ.get("SJASMPLUS") or shutil.which("sjasmplus") or "sjasmplus"
STOP = 0x5B00          # ловушка возврата: буфер принтера, программа его не трогает


def assemble(src: Path, out_dir: Path, defines: dict | None = None) -> tuple[bytes, int, dict]:
    """-> (байты от ORG до конца, ORG, символы)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = out_dir / (src.stem + ".bin")
    sym = out_dir / (src.stem + ".sym")
    cmd = [str(SJASM), "--nologo", "--msg=war", f"--raw={raw}", f"--sym={sym}", "--fullpath"]
    for k, v in (defines or {}).items():
        cmd.append(f"-D{k}={v}")
    cmd.append(src.as_posix())
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=src.parent)
    if r.returncode != 0:
        raise RuntimeError(r.stdout + r.stderr)
    syms = {}
    for line in sym.read_text().splitlines():
        m = re.match(r"(\S+):\s+EQU\s+0x([0-9A-Fa-f]+)", line)
        if m:
            syms[m.group(1)] = int(m.group(2), 16)
    return raw.read_bytes(), syms["ORG_START"], syms


class Machine:
    def __init__(self, code: bytes, org: int, syms: dict):
        mem = [0] * 65536
        rom = read_bin_file(ROM48)
        mem[:len(rom)] = rom
        mem[org:org + len(code)] = list(code)
        self.sim = from_memory(CSimulator, mem, {"PC": org, "SP": 0xFF00}, {"iff": 0, "im": 1},
                               {"fast_djnz": False, "fast_ldir": False})
        self.mem = self.sim.memory
        self.reg = self.sim.registers
        self.syms = syms

    def sym(self, name):
        return self.syms[name]

    def poke(self, addr, data):
        for i, b in enumerate(data):
            self.mem[addr + i] = b & 0xFF

    def peek(self, addr, n) -> bytes:
        return bytes(self.mem[addr:addr + n])

    def poke_words(self, addr, values):
        for i, v in enumerate(values):
            v &= 0xFFFF
            self.mem[addr + 2 * i] = v & 0xFF
            self.mem[addr + 2 * i + 1] = v >> 8

    def peek_words(self, addr, n, signed=True):
        out = []
        for i in range(n):
            v = self.mem[addr + 2 * i] | (self.mem[addr + 2 * i + 1] << 8)
            if signed and v >= 0x8000:
                v -= 0x10000
            out.append(v)
        return out

    def peek_s8(self, addr, n):
        return [b - 256 if b >= 128 else b for b in self.peek(addr, n)]

    def call(self, routine, sp=0xFF00, max_t=10 ** 10, **regs) -> int:
        """Вызвать подпрограмму, вернуть число тактов."""
        if isinstance(routine, str):
            routine = self.sym(routine)
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
        self.sim.run(routine, STOP)
        return r[T] - t0

    def regpair(self, name):
        hi, lo = {"bc": (B, C), "de": (D, E), "hl": (H, L)}[name]
        return (self.reg[hi] << 8) | self.reg[lo]
