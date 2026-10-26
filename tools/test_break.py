"""BREAK посреди истории: снимок после ленты (build/hippo.z80) -> затравка -> несколько токенов ->
CAPS SHIFT + SPACE посреди токена -> программа сразу у ввода -> новая затравка пишется как обычно.

    python tools/test_break.py
Экраны: build/break_writing.png (пишет: часы и "Writing..."), build/break_after.png (после второй затравки).
"""
from __future__ import annotations

import sys
from pathlib import Path

from skoolkit.simutils import PC

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_tap import Run, chord  # noqa: E402
from test_story128 import screen_png  # noqa: E402
from zx128 import BUILD, load_syms  # noqa: E402

ICON_OFF = 0x1000 + 7 * 32           # знакоместо (23, 0): смещение в банке 5


def main():
    S = load_syms()
    m = Run(BUILD / "hippo.z80")
    bank5 = m.simulator.memory.banks[5]
    var = lambda a, n=1: bytes(bank5[a - 0x4000:a - 0x4000 + n])
    gen_n = lambda: int.from_bytes(var(S["GEN_N"], 2), "little")
    icon = lambda: bytes(bank5[ICON_OFF + 256 * k] for k in range(8))
    in_input = lambda: S["scan_key"] <= m.simulator.registers[PC] < S["space_pressed"]
    shot = lambda name: screen_png(bytes(bank5[:6912]), BUILD / name)

    def typeline(text):
        for ch in text + "\n":
            m.press(chord(ch))
            m.run(400_000)
            m.press([])
            m.run(400_000)

    m.run(3_000_000)
    assert in_input(), "не дошли до ввода"
    idle = icon()                             # в покое там начало строки версии
    typeline("Once upon a time")
    frames = set()
    while gen_n() < 2:                        # затравка (5 прогонов) и пара токенов; кадры часов — по ходу
        m.run(1_000_000)
        frames.add(icon())
    m.run(5_500_000)                          # посреди очередного токена
    shot("break_writing.png")
    n_before = gen_n()
    print(f"пишет: токенов {n_before}, разных кадров часов {len(frames)}")
    assert len(frames) >= 3, "часы не крутятся"

    m.press(["CS", "SPACE"])                  # BREAK
    t = 0
    while not in_input() and t < 14_000_000:
        m.run(50_000)
        t += 50_000
        if t == 700_000:                      # отпустить через ~0,2 с
            m.press([])
    m.press([])
    m.run(200_000)
    print(f"BREAK: у ввода через {t / 3.5e6:.2f} с, токенов {gen_n()} (было {n_before}), "
          f"часы стёрты: {icon() == idle}")
    assert in_input() and t <= 1_000_000, "BREAK не сработал сразу"
    assert icon() == idle, "часы не стёрты"
    assert var(S["PROMPT_LEN"]) == b"\0", "в новую затравку что-то попало"

    typeline("The little Speccy")
    for _ in range(300):
        m.run(1_000_000)
        if gen_n() >= 3:
            break
    print(f"вторая затравка: токенов {gen_n()}")
    assert gen_n() > 0
    shot("break_after.png")
    print("экраны:", BUILD / "break_writing.png", BUILD / "break_after.png")


if __name__ == "__main__":
    main()
