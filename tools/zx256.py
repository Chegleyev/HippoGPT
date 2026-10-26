"""Стенд Z80 для Scorpion ZS-256: шестнадцать страниц по 16 КБ на C000.

Страница = биты 0..2 порта #7FFD + бит 4 порта #1FFD (страницы 8..15). Остальные биты #1FFD программа не
трогает, стенд их не моделирует.

Ядро SkoolKit на C знает только восемь банков 128K и само переключает их по #7FFD (запись в #1FFD для него —
та же #7FFD, как на 128K: программа после #1FFD всегда пишет и #7FFD). Поэтому страницы 8..15 стенд
подставляет содержимым: при смене бита 4 #1FFD буферы банков 0,1,3,4,6,7 меняются данными со страницами
8,9,11,12,14,15. Страницы 10 и 13 так не изобразить (их буферы — неподвижные банки 2 и 5 на 8000 и 4000):
сборка их не занимает, а обращение к ним стенд считает ошибкой.
"""
from __future__ import annotations

from pathlib import Path

from skoolkit import CSimulator
from skoolkit.pagingtracer import Memory
from skoolkit.simutils import from_memory

from zx128 import ROOT, Machine128, load_syms

BUILD = ROOT / "build256"
SWAP = (0, 1, 3, 4, 6, 7)


class Machine256(Machine128):
    """Machine128 с памятью Scorpion: build256/bank0..15.bin, порты #7FFD и #1FFD."""

    def __init__(self, syms=None, screen: bytes | None = None, build: Path = BUILD):
        pages = []
        for n in range(16):
            p = build / f"bank{n}.bin"
            pages.append(bytearray(p.read_bytes()) if p.exists() else bytearray(16384))
        for n in (10, 13):
            assert not any(pages[n]), f"страница {n} занята — стенд её не изобразит"
        if screen is not None:
            pages[5][:len(screen)] = screen
        self.store = pages                      # «дома» страниц 8..15 (и 0..7, пока на их месте верхние)
        mem = Memory([list(pages[n]) for n in range(8)], 0x10, "128K")
        self.simulator = from_memory(CSimulator, mem, {"PC": 0, "SP": 0x5E00}, {"iff": 0, "im": 1}, {})
        self.out7ffd = 0x10
        self.outfffd = 0
        self.ay = [0] * 16
        self.outfe = 0
        self.border = 0
        self.simulator.set_tracer(self)
        self.mem = self.simulator.memory
        self.reg = self.simulator.registers
        self.syms = syms or load_syms(build / "hippo.sym")
        self.keyboard = [0] * 8
        self.hi = 0                             # бит 4 #1FFD: на месте банков SWAP — страницы +8
        self.switches = 0

    def _banks(self):
        return self.simulator.memory.banks

    def set_hi(self, hi):
        if hi == self.hi:
            return
        banks = self._banks()
        for k in SWAP:
            self.store[k + 8 * self.hi][:] = banks[k]
            banks[k][:] = self.store[k + 8 * hi]
        self.hi = hi
        self.switches += 1

    def write_port(self, registers, port, value, offset):
        if port & 0xFF == 0xFD and port >> 8 == 0x1F:      # #1FFD
            self.set_hi(1 if value & 0x10 else 0)
        elif port & 0x8002 == 0 and self.hi and (value & 7) in (2, 5):
            raise RuntimeError(f"страница {8 + (value & 7)} на стенде не изображается")
        super().write_port(registers, port, value, offset)

    def bank(self, n):
        """Подключить страницу n (0..15) на C000 (для чтения из стенда)."""
        assert n not in (10, 13)
        self.set_hi(1 if n & 8 else 0)
        v = 0x10 | (n & 7)
        self.simulator.memory.out7ffd(v)
        self.out7ffd = v
