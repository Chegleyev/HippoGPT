"""Сборка HippoGPT 128K: целочисленная модель -> include-файлы -> sjasmplus -> образы банков.

    python tools/build128.py models/hippo128.pt      (или runs/<прогон>/last.pt после обучения, или build/intmodel128.pkl)

Раскладка памяти — README, раздел «Память». Пишет z80/gen_*.inc|asm и build/bank{0..7}.bin, build/hippo.sym.
"""
from __future__ import annotations

import os
import pickle
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ref"))
sys.path.insert(0, str(ROOT / "tools"))
import intref            # noqa: E402
import gen_matvec4       # noqa: E402
import gen_matvec4n      # noqa: E402
import sampler           # noqa: E402
import json              # noqa: E402

TEMP = 0.6               # температура по умолчанию; на Спектруме цифры 1..9 -> 0.1..0.9, 0 -> 1.0
MIN_EOS = 32             # сказка не кончается раньше стольких токенов (иначе бывает одна фраза)
from zx import SJASM     # noqa: E402

Z80 = ROOT / "z80"
BUILD = ROOT / "build"

# Python с torch для ref/export.py (SkoolKit и сборка работают и без него)
TORCH_PY = os.environ.get("HIPPO_TORCH_PY", sys.executable)


def db_lines(data, per_line=16):
    data = [int(x) & 0xFF for x in data]
    return [("        db " + ",".join(str(b) for b in data[i:i + per_line])) for i in range(0, len(data), per_line)]


def gen_dot(n: int, name: str) -> list[str]:
    """Сумма SQ[(DE+i) - imm_i], i < n -> IXL:B:C. Строка не переходит границу страницы."""
    o = [f"{name}:", "        ld h,SQLO >> 8", "        ld bc,0", "        ld ixl,c"]
    for i in range(n):
        o += [f"{name}_b{i}:",
              "        ld a,(de)",
              "        inc e",
              "        sub 0",
              "        ld l,a",
              "        ld a,(hl)",
              "        add a,c",
              "        ld c,a",
              "        inc h",
              "        ld a,(hl)",
              "        adc a,b",
              "        ld b,a",
              "        dec h",
              "        jr nc,$+4",
              "        inc ixl"]
    o += ["        ret", f"{name}_imm equ {name}_b0 + 3"]
    return o

LD_FIELDS = [("WQKV", 2), ("WQKV_PG", 1), ("MQKV", 2), ("SQKV", 1), ("WO", 2), ("WO_PG", 1), ("MO", 2),
             ("SO", 1), ("W1", 2), ("W1_PG", 1), ("M1", 2), ("S1", 1), ("W2", 2), ("W2_PG", 1), ("M2", 2),
             ("S2", 1), ("BIAS", 2), ("ACT", 2), ("KC", 2), ("AK", 2), ("VT", 2), ("V2", 2)]

# где лежат матрицы: (банк, метка)
PLACE = {
    0: dict(WQKV=6, WO=6, W1=0, W2=3),
    1: dict(WQKV=6, WO=7, W1=1, W2=3),
}
EMB_PG = 4
# дополнительные поля описателя слоя (сборка 256K): [(имя, размер, функция (номер слоя) -> значение)]
EXTRA_FIELDS = []


def write_sources(im: intref.IntModel):
    assert im.emb_mul is not None, "128K-сборка рассчитана на 4-битные вложения"
    L = len(im.layers)
    qkv_rows = im.heads * im.dh + 2 * im.dh
    p = ["; сгенерировано tools/build128.py",
         f"VOCAB equ {im.vocab}", f"D_MODEL equ {im.d}", f"N_HEADS equ {im.heads}", f"D_HEAD equ {im.dh}",
         f"QDIM equ {im.heads * im.dh}", f"QKV_ROWS equ {qkv_rows}", f"D_FFN equ {im.ffn}", f"CTX equ {im.ctx}",
         f"N_LAYERS equ {L}", f"FINAL_SHIFT equ {im.final_shift}", f"FINAL_LIM equ {im.final_lim}",
         "DOT_STRIDE equ 17", f"EMB_PG equ {EMB_PG}",
         # версия 128K: без входных таблиц и страниц 8..15; законченное слово затравки не склеивается
         "SCORPION equ 0", "INPUT_TABLES equ 0", "PREV_EMB equ 0", "BIG_ROWS equ 0", "HEAL_WORDS equ 0",
         "PIN equ 0", f"MIN_EOS equ {MIN_EOS}"]
    lm, unit_z = sampler.logit_mults(im.logit_row)
    p.append(f"TQ_DEFAULT equ {sampler.temp_q(unit_z, TEMP)}")
    key = round(TEMP * 10) % 10
    assert abs(TEMP - (key or 10) / 10) < 1e-9, "температура по умолчанию должна быть на клавише"
    p.append(f"TEMP_KEY_DEFAULT equ '{key}'")
    for k in range(10):
        p.append(f"TQ_K{k} equ {sampler.temp_q(unit_z, (k or 10) / 10)}")
    off = 0
    for name, size in LD_FIELDS + [(n, sz) for n, sz, _ in EXTRA_FIELDS]:
        p.append(f"LD_{name} equ {off}")
        off += size
    p.append(f"LAYER_SIZE equ {off}")
    (Z80 / "gen_params.inc").write_text("\n".join(p) + "\n", encoding="utf-8")

    (Z80 / "gen_exp63.inc").write_text("\n".join(db_lines(intref.EXP63)) + "\n")
    (Z80 / "gen_exp127.inc").write_text("\n".join(db_lines(sampler.EXP127)) + "\n")
    rs8 = [(int(k) + 128) >> 8 for k in intref.RSQRT]
    (Z80 / "gen_rsq8.inc").write_text("\n".join(db_lines(rs8)) + "\n")
    act = []
    for ly in im.layers:
        act += list(ly.act)
    (Z80 / "gen_act.inc").write_text("\n".join(db_lines(act)) + "\n")

    (Z80 / "gen_matvec4.asm").write_text(gen_matvec4.gen("break_now"), encoding="utf-8")
    (Z80 / "gen_matvec4n.asm").write_text(gen_matvec4n.gen("break_now"), encoding="utf-8")
    dot = ["; сгенерировано tools/build128.py"] + gen_dot(16, "dot16") + gen_dot(64, "dot64")
    (Z80 / "gen_dot.asm").write_text("\n".join(dot) + "\n", encoding="utf-8")

    # описатели слоёв и множители строк — неподвижная память (банк 5)
    m = ["; сгенерировано tools/build128.py: описатели слоёв и множители строк", "LAYERS:"]
    for li, ly in enumerate(im.layers):
        pl = PLACE[li]
        f = dict(WQKV=f"L{li}_WQKV", WQKV_PG=pl["WQKV"], MQKV=f"L{li}_MQKV", SQKV=ly.sqkv,
                 WO=f"L{li}_WO", WO_PG=pl["WO"], MO=f"L{li}_MO", SO=ly.so,
                 W1=f"L{li}_W1", W1_PG=pl["W1"], M1=f"L{li}_M1", S1=ly.s1,
                 W2=f"L{li}_W2", W2_PG=pl["W2"], M2=f"L{li}_M2", S2=ly.s2,
                 BIAS=f"BIAS{li}", ACT=f"ACTTAB + {128 * li}", KC=f"KC{li}", AK=f"AK{li}", VT=f"VT{li}", V2=f"V2{li}")
        for name, size in LD_FIELDS:
            m.append(f"        {'dw' if size == 2 else 'db'} {f[name]}")
        for name, size, fn in EXTRA_FIELDS:
            m.append(f"        {'dw' if size == 2 else 'db'} {fn(li)}")
    for li, ly in enumerate(im.layers):
        for nm, mul in [("MQKV", ly.mqkv), ("MO", ly.mo), ("M1", ly.m1), ("M2", ly.m2)]:
            m.append(f"L{li}_{nm}:")
            m += db_lines(mul)
    (Z80 / "gen_mults.inc").write_text("\n".join(m) + "\n", encoding="utf-8")

    # банки на C000
    b = ["; сгенерировано tools/build128.py: данные по банкам"]

    def bank(n):
        b.extend([f"        page {n}", "        org 0xC000"])

    def blob(label, data):
        b.append(f"{label}:")
        b.extend(db_lines(data))

    l0, l1 = im.layers
    bank(0)
    blob("L0_W1", gen_matvec4.pack_rows(l0.t1))
    bank(1)
    blob("L1_W1", gen_matvec4.pack_rows(l1.t1))
    bank(3)
    blob("L0_W2", gen_matvec4n.pack_rows(l0.t2))
    blob("L1_W2", gen_matvec4n.pack_rows(l1.t2))
    bank(EMB_PG)
    blob("EMB", gen_matvec4n.pack_rows(im.emb))
    bank(6)
    blob("L0_WQKV", gen_matvec4.pack_rows(l0.tq))
    blob("L0_WO", gen_matvec4.pack_rows(l0.to))
    blob("L1_WQKV", gen_matvec4.pack_rows(l1.tq))
    bank(7)
    if PLACE[1]["WO"] == 7:             # сборка 256K кладёт его на другую страницу сама
        blob("L1_WO", gen_matvec4.pack_rows(l1.to))
    b += ["        org 0xD000", "KV_ZERO:",
          "KC0:    ds CTX * D_HEAD", "KC1:    ds CTX * D_HEAD",
          "VT0:    ds CTX * D_HEAD", "VT1:    ds CTX * D_HEAD",
          "AK0:    ds CTX * 2", "AK1:    ds CTX * 2",
          "V20:    ds D_HEAD * 3", "V21:    ds D_HEAD * 3",
          "KV_ZERO_LEN equ $ - KV_ZERO",
          "        assert (KC0 & 15) == 0 && (VT0 & 63) == 0 && (VT1 & 63) == 0",
          "        org 0xE200"]
    for li, ly in enumerate(im.layers):
        assert ly.bias.min() >= -128 and ly.bias.max() <= 127
        blob(f"BIAS{li}", ly.bias.reshape(-1))
    assert im.emb_mul.max() < 256
    blob("EMBMUL", im.emb_mul)
    blob("LOGMUL", lm)
    # строки токенов: таблица смещений + строки «длина, символы»; порядок для жадного разбора
    meta = json.loads((ROOT / "data" / f"{im.data}.json").read_text(encoding="utf-8"))
    pieces = meta["pieces"]
    assert len(pieces) == im.vocab
    b.append("TOKOFF:")
    for t in range(im.vocab):
        b.append(f"        dw TOKSTR_{t}")
    b.append("TOKSTR:")
    for t, pc in enumerate(pieces):
        body = [0] if t == 0 else [len(pc)] + [ord(c) for c in pc]
        b.append(f"TOKSTR_{t}: db " + ",".join(str(x) for x in body))
    # разбор затравки слияниями BPE: пары (a, b) по номерам (токен слияния = 64 + номер), символ -> токен
    merges = meta["merges"]
    assert len(merges) == im.vocab - len(sampler.VOCAB)
    b.append("MERGES:")
    for i in range(0, len(merges), 8):
        b.append("        dw " + ",".join(f"{a},{c}" for a, c in merges[i:i + 8]))
    charid = [sampler.VOCAB.index(chr(c)) if chr(c) in sampler.VOCAB[1:] else 2 for c in range(128)]
    b.append("CHARID:")
    b.extend(db_lines(charid))
    b.append("GENLOG: ds 2 * 256")
    b.append("BANK7_END:")
    b.append("        assert BANK7_END <= 0x10000")
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
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else BUILD / "intmodel128.pkl"
    pkl = BUILD / "intmodel128.pkl"
    if src.suffix == ".pt":
        subprocess.run([TORCH_PY, str(ROOT / "ref" / "export.py"), str(src), str(pkl)], check=True)
    else:
        pkl = src
    with open(pkl, "rb") as f:
        im = pickle.load(f)
    write_sources(im)
    syms = assemble()
    print("start", hex(syms["start"]), "forward", hex(syms["forward"]))


if __name__ == "__main__":
    main()
