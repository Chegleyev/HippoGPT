"""Сборка HippoGPT для Scorpion ZS-256: как tools/build128.py, плюс страницы 8..15.

    python tools/build256.py models/hippo256.pt      (или runs/<прогон>/last.pt после обучения, или build256/intmodel256.pkl)

Страницы 0..7 — раскладка 128K (README, раздел «Память»). Дальше:
    8          таблица предыдущего токена (512 строк x 64 кода по два в байт)
    9          хеши пары BIGH1/BIGH2, множители строк PREVMUL и BIGMUL
    11,12,14,15 таблица пар: 512 строк на страницу (номера — в BIGPG на странице 9; 10 и 13 свободны:
               стенд zx256 их не изображает)
Пишет z80/gen_*.inc|asm и build256/bank{0..15}.bin, build256/hippo.sym.
"""
from __future__ import annotations

import pickle
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ref"))
sys.path.insert(0, str(ROOT / "tools"))
import build128          # noqa: E402
import gen_matvec4n      # noqa: E402
from build128 import db_lines, TORCH_PY  # noqa: E402
from zx import SJASM     # noqa: E402

Z80 = ROOT / "z80"
BUILD = ROOT / "build256"
PREV_PG, TAB_PG = 8, 9
BIG_PAGES = [11, 12, 14, 15]


def write_sources(im):
    pin = getattr(im, "pin", 0)
    if pin:
        # закреплённые: веса и выход идут через dot16 — ровно 16 позиций по 16 чисел
        assert pin == 16 and im.dh == 16
        build128.EXTRA_FIELDS[:] = [(n, 2, (lambda li, n=n: f"{n}{li}")) for n in ("KCP", "AKP", "VTP", "V2P", "PINB")]
    else:
        build128.EXTRA_FIELDS[:] = []
    build128.PLACE[1]["WO"] = TAB_PG if pin else 7     # страница 7 занята почти вся: Wo слоя 1 — к хешам
    try:
        build128.write_sources(im)
    finally:
        build128.PLACE[1]["WO"] = 7
    prev, big = im.prev is not None, im.big is not None
    rows = len(im.big) if big else 0
    assert rows // 512 <= len(BIG_PAGES), "таблица пар не влезает в страницы 11, 12, 14, 15"
    par = (Z80 / "gen_params.inc").read_text(encoding="utf-8")
    flags = {"SCORPION": 1, "INPUT_TABLES": int(prev or big), "PREV_EMB": int(prev), "BIG_ROWS": rows,
             "HEAL_WORDS": 0, "PIN": pin, "MIN_EOS": build128.MIN_EOS}
    for k, v in flags.items():
        par, n = re.subn(rf"^{k} equ .*$", f"{k} equ {v}", par, flags=re.M)
        assert n == 1, k
    par += f"PREV_PG equ {PREV_PG}\nTAB_PG equ {TAB_PG}\n"
    (Z80 / "gen_params.inc").write_text(par, encoding="utf-8")

    b = [(Z80 / "gen_banks.asm").read_text(encoding="utf-8"), "; страницы Scorpion (tools/build256.py)"]

    def page(n):
        b.extend([f"        page {n}", "        org 0xC000"])

    def blob(label, data):
        b.append(f"{label}:")
        b.extend(db_lines(data))

    if pin:
        # закреплённые — в банке 7 за строками (там же, где кэш; внимание работает при банке 7)
        b += ["        page 7", "        org 0xC000", "PIN_ZERO:"]
        for li in range(len(im.layers)):
            b += [f"KCP{li}:   ds PIN * D_HEAD", f"AKP{li}:   ds PIN * 2", f"VTP{li}:   ds D_HEAD * PIN",
                  f"V2P{li}:   ds D_HEAD * 3"]
        b.append("PIN_ZERO_LEN equ $ - PIN_ZERO")
        for li, ly in enumerate(im.layers):
            assert ly.pin_bias.min() >= -128 and ly.pin_bias.max() <= 127
            blob(f"PINB{li}", ly.pin_bias)
        b += ["PIN_END:", "        assert PIN_END <= KV_ZERO"]
    if prev:
        assert im.prev_mul.max() < 256
        page(PREV_PG)
        blob("PREVTAB", gen_matvec4n.pack_rows(im.prev))
    page(TAB_PG)
    if pin:
        import gen_matvec4
        blob("L1_WO", gen_matvec4.pack_rows(im.layers[1].to))
    if big:
        assert im.big_mul.max() < 256 and rows % 512 == 0
        b.append("BIGH1:")
        b.extend(f"        dw {int(v)}" for v in im.h1)
        b.append("BIGH2:")
        b.extend(f"        dw {int(v)}" for v in im.h2)
        blob("BIGMUL", im.big_mul)
        blob("BIGPG", BIG_PAGES[:rows // 512])
    if prev:
        blob("PREVMUL", im.prev_mul)
    b.append("TAB_END:")
    b.append("        assert TAB_END <= 0x10000")
    if big:
        packed = gen_matvec4n.pack_rows(im.big)
        for k in range(rows // 512):
            page(BIG_PAGES[k])
            blob(f"BIGTAB{k}", packed[k * 16384:(k + 1) * 16384])
    (Z80 / "gen_banks.asm").write_text("\n".join(b) + "\n", encoding="utf-8")


def assemble():
    BUILD.mkdir(exist_ok=True)
    sym = BUILD / "hippo.sym"
    cmd = [str(SJASM), "--nologo", "--msg=war", f"--sym={sym}", "--fullpath", (Z80 / "hippo.asm").as_posix()]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=BUILD)
    if r.returncode != 0:
        raise RuntimeError(r.stdout + r.stderr)
    syms = {}
    for line in sym.read_text().splitlines():
        mm = re.match(r"(\S+):\s+EQU\s+0x([0-9A-Fa-f]+)", line)
        if mm:
            syms[mm.group(1)] = int(mm.group(2), 16)
    print(r.stdout.strip())
    return syms


def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else BUILD / "intmodel256.pkl"
    pkl = BUILD / "intmodel256.pkl"
    BUILD.mkdir(exist_ok=True)
    if src.suffix == ".pt":
        subprocess.run([TORCH_PY, str(ROOT / "ref" / "export.py"), str(src), str(pkl)], check=True)
    else:
        pkl = src
    with open(pkl, "rb") as f:
        im = pickle.load(f)
    write_sources(im)
    syms = assemble()
    print("start", hex(syms["start"]), "forward", hex(syms["forward"]),
          "банк 5 до", hex(syms["BANK5_END"]), "банк 2 до", hex(syms["HOT_END"]))


if __name__ == "__main__":
    main()
