"""Сверка ядра «по два веса в байт» (ptn_build + matvec4n) с numpy и замер тактов."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen_matvec4n
from zx import ROOT, Machine, assemble

XIN, OUT = 0x7000, 0x7400
STACK = 0x6F00


def main():
    (ROOT / "z80" / "gen_matvec4n.asm").write_text(gen_matvec4n.gen(), encoding="utf-8")
    (ROOT / "z80" / "test_matvec4n.asm").write_text(
        "        org 0x5E00\nORG_START:\nPTN equ 0x8000\n        include \"gen_matvec4n.asm\"\n", encoding="utf-8")
    code, org, syms = assemble(ROOT / "z80" / "test_matvec4n.asm", ROOT / "build")
    print("размер кода", len(code))
    rng = np.random.default_rng(1)
    for din, dout, xmax in [(256, 64, 127), (64, 512, 63), (64, 8, 504), (32, 4, 1000)]:
        m = Machine(code, org, syms)
        c = rng.integers(-7, 8, size=(dout, din))
        x = rng.integers(-xmax, xmax + 1, size=din)
        w = gen_matvec4n.pack_rows(c)
        base = 0xC000
        m.poke(base, w)
        m.poke_words(XIN, x)
        t_pt = m.call("ptn_build", sp=STACK, b=din & 255, ix=XIN)
        m.poke(syms["mv4n_g"] + 2, [din // gen_matvec4n.GROUP])
        t_mv = 0
        for r0 in range(0, dout, 256):                    # за вызов — не больше 256 строк
            rows = min(256, dout - r0)
            t_mv += m.call("matvec4n", sp=STACK, de=base + r0 * din // 2, ix=OUT + 2 * r0, iy=(rows & 255) << 8)
        got = np.array(m.peek_words(OUT, dout))
        ref = c @ x
        ref16 = ((ref + 32768) % 65536) - 32768
        ok = np.array_equal(got, ref16)
        print(f"{din:4d}x{dout:<4d} ok={ok}  таблица {t_pt / din:.0f} тактов на вход, "
              f"matvec {t_mv / (din * dout):.1f} на вес")
        if not ok:
            bad = np.nonzero(got != ref16)[0]
            print("  ", bad[:8], got[bad[:4]], ref16[bad[:4]])
            sys.exit(1)


if __name__ == "__main__":
    main()
