"""Продолжить снимок после загрузки ленты (build/hippo.z80): набрать затравку на клавиатуре Спектрума,
дать Хиппо написать N токенов, снять экран.

    python tools/run_tap.py "Once upon a time" [токенов]       (цифра в затравке — смена температуры)

Сначала сверяет загруженную память с образами банков — так проверяется сама лента.
"""
from __future__ import annotations

import sys
from pathlib import Path

from skoolkit import CSimulator
from skoolkit.pagingtracer import Memory, PagingTracer
from skoolkit.snapshot import Snapshot
from skoolkit.simutils import from_snapshot, PC, T

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_story128 import screen_png  # noqa: E402
from zx128 import BUILD, load_syms  # noqa: E402

KEYS = {
    "CS": (0, 0x01), "z": (0, 0x02), "x": (0, 0x04), "c": (0, 0x08), "v": (0, 0x10),
    "a": (1, 0x01), "s": (1, 0x02), "d": (1, 0x04), "f": (1, 0x08), "g": (1, 0x10),
    "q": (2, 0x01), "w": (2, 0x02), "e": (2, 0x04), "r": (2, 0x08), "t": (2, 0x10),
    "1": (3, 0x01), "2": (3, 0x02), "3": (3, 0x04), "4": (3, 0x08), "5": (3, 0x10),
    "0": (4, 0x01), "9": (4, 0x02), "8": (4, 0x04), "7": (4, 0x08), "6": (4, 0x10),
    "p": (5, 0x01), "o": (5, 0x02), "i": (5, 0x04), "u": (5, 0x08), "y": (5, 0x10),
    "ENTER": (6, 0x01), "l": (6, 0x02), "k": (6, 0x04), "j": (6, 0x08), "h": (6, 0x10),
    "SPACE": (7, 0x01), "SS": (7, 0x02), "m": (7, 0x04), "n": (7, 0x08), "b": (7, 0x10),
}
SYM = {".": "m", ",": "n", "!": "1", "?": "c", "'": "7", '"': "p", "-": "j", ":": "z", ";": "o"}


def chord(ch):
    if ch == " ":
        return ["SPACE"]
    if ch == "\n":
        return ["ENTER"]
    if ch in SYM:
        return ["SS", SYM[ch]]
    if ch.isupper():
        return ["CS", ch.lower()]
    return [ch]


class Run(PagingTracer):
    def __init__(self, path):
        snap = Snapshot.get(str(path))
        self.simulator = from_snapshot(CSimulator, snap, {}, {}, {"fast_djnz": False, "fast_ldir": False}, None)
        if isinstance(self.simulator.memory, Memory):
            self.simulator.memory.out7ffd(snap.out7ffd)
        self.out7ffd = snap.out7ffd
        self.outfffd = snap.outfffd
        self.ay = list(snap.ay)
        self.outfe = snap.outfe
        self.border = snap.border
        self.simulator.set_tracer(self)
        self.keyboard = [0] * 8

    def read_port(self, registers, port):
        """Клавиатура: чётный порт, полуряды выбираются нулями старшего байта."""
        if port % 2 == 0:
            result = 0xFF
            high = port >> 8
            for row in range(8):
                if not high & (1 << row):
                    result &= ~self.keyboard[row] & 0xFF
            return result
        return 0xFF

    def press(self, keys):
        self.keyboard = [0] * 8
        for k in keys:
            row, bit = KEYS[k]
            self.keyboard[row] |= bit

    def run(self, t):
        r = self.simulator.registers
        self.simulator.trace(r[PC], 0x10000, 0, r[T] + t, False, None, None, self.keyboard, None, None)


def main():
    prompt = sys.argv[1] if len(sys.argv) > 1 else "Once upon a time"
    ntok = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    syms = load_syms()
    m = Run(BUILD / "hippo.z80")
    mem = m.simulator.memory
    # 1. загруженное == образы банков
    ok = True
    for n in range(8):
        img = (BUILD / f"bank{n}.bin").read_bytes()
        got = bytes(mem.banks[n])
        if n == 5:
            lo, hi = syms["ORG_START"] - 0x4000, syms["BANK5_END"] - 0x4000
        elif n == 2:
            lo, hi = syms["EXP63"] - 0x8000, syms["HOT_END"] - 0x8000
        else:
            lo, hi = 0, len(img.rstrip(b"\0"))
        same = got[lo:hi] == img[lo:hi]
        ok &= same
        print(f"банк {n}: {'совпадает' if same else 'НЕ СОВПАДАЕТ'} ({hi - lo} байт)")
    if not ok:
        sys.exit(1)
    # 2. запуск: дойти до ввода, набрать затравку, ENTER
    m.run(3_000_000)
    for ch in prompt + "\n":
        m.press(chord(ch))
        m.run(400_000)
        m.press([])
        m.run(400_000)
    m.run(ntok * 15_000_000)
    tk = chr(mem.banks[5][syms["TEMP_KEY"] - 0x4000])
    tq = mem.banks[5][syms["TQ"] - 0x4000] + 256 * mem.banks[5][syms["TQ"] + 1 - 0x4000]
    print(f"температура: клавиша {tk}, TQ = {tq}")
    scr = bytes(mem.banks[5][:6912])
    screen_png(scr, BUILD / "tap_run.png")
    print("экран:", BUILD / "tap_run.png")


if __name__ == "__main__":
    main()
