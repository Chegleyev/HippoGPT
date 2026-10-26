"""Сверка 4-битного ядра (таблица кратных + matvec4) с numpy и замер тактов."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen_matvec4
from zx import ROOT, Machine, assemble

XIN, OUT = 0xB000, 0xB400
STACK = 0xAF00


def main():
    (ROOT / "z80" / "gen_matvec4.asm").write_text(gen_matvec4.gen(), encoding="utf-8")
    (ROOT / "z80" / "test_matvec4.asm").write_text(
        "        org 0x8000\nORG_START:\nPT equ 0x6000\n        include \"gen_matvec4.asm\"\n", encoding="utf-8")
    code, org, syms = assemble(ROOT / "z80" / "test_matvec4.asm", ROOT / "build")
    rng = np.random.default_rng(0)
    for din, dout, xmax in [(64, 96, 127), (64, 64, 504), (64, 256, 127), (256, 64, 127)]:
        m = Machine(code, org, syms)
        c = rng.integers(-7, 8, size=(dout, din))
        x = rng.integers(-xmax, xmax + 1, size=din)
        w = gen_matvec4.pack_rows(c)
        base = 0xC000                                  # выровнено под строки любой длины
        m.poke(base, w)
        m.poke_words(XIN, x)
        t_pt = m.call("pt_build", sp=STACK, b=din & 255, ix=XIN)
        m.poke(syms["mv4_n"] + 2, [din // gen_matvec4.BLOCK])
        t_mv = m.call("matvec4", sp=STACK, de=base, ix=OUT, iy=(dout & 255) << 8)
        got = np.array(m.peek_words(OUT, dout))
        ref = c @ x
        ref16 = ((ref + 32768) % 65536) - 32768
        ok = np.array_equal(got, ref16)
        print(f"{din:4d}x{dout:<4d} ok={ok}  таблица {t_pt} тактов ({t_pt / din:.0f} на вход), "
              f"matvec {t_mv} ({t_mv / (din * dout):.1f} на вес), |сумма|max {np.abs(ref).max()}")
        if not ok:
            bad = np.nonzero(got != ref16)[0]
            print("  ", bad[:8], got[bad[:4]], ref16[bad[:4]])
            sys.exit(1)


if __name__ == "__main__":
    main()
