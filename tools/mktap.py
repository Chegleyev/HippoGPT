"""Лента HippoGPT для 128K: build/hippo.tap.

Блоки: Бейсик «HippoGPT» (автозапуск) -> SCREEN$ с шапкой -> загрузчик в машинных кодах -> блоки без
заголовков, которые загрузчик через LD-BYTES ПЗУ раскладывает по банкам:
    10 CLEAR 24063: LOAD "" CODE : LOAD "" SCREEN$ : RANDOMIZE USR 24064
Загрузчик — до картинки: так сообщения ПЗУ «Bytes: ...» затирает сама картинка.
Загрузчик живёт в 5E00 (там потом буферы программы) и в конце прыгает на start.
"""
from __future__ import annotations

import struct
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from zx import SJASM  # noqa: E402
from zx128 import BUILD, ROOT, load_syms  # noqa: E402

LOADER_ORG = 0x5E00
CLEAR_AT = LOADER_ORG - 1


def tap_block(flag: int, data: bytes) -> bytes:
    body = bytes([flag]) + data
    chk = 0
    for b in body:
        chk ^= b
    body += bytes([chk])
    return struct.pack("<H", len(body)) + body


def header(kind: int, name: str, length: int, p1: int, p2: int) -> bytes:
    nm = name.encode("ascii")[:10].ljust(10, b" ")
    return tap_block(0x00, bytes([kind]) + nm + struct.pack("<HHH", length, p1, p2))


def bas_num(n: int) -> bytes:
    """Число в строке Бейсика: цифры + 0E + пятибайтовое целое."""
    return str(n).encode() + bytes([0x0E, 0, 0, n & 255, n >> 8, 0])


def basic_loader() -> bytes:
    CLEAR, LOAD, SCREEN, CODE, RAND, USR = 0xFD, 0xEF, 0xAA, 0xAF, 0xF9, 0xC0
    line = (bytes([CLEAR]) + bas_num(CLEAR_AT) + b":" + bytes([LOAD]) + b'""' + bytes([CODE]) + b":"
            + bytes([LOAD]) + b'""' + bytes([SCREEN]) + b":" + bytes([RAND, USR]) + bas_num(LOADER_ORG) + b"\r")
    return struct.pack(">H", 10) + struct.pack("<H", len(line)) + line


def chunks(syms):
    """(банк, адрес, байты) — что грузить."""
    out = []
    b5 = (BUILD / "bank5.bin").read_bytes()
    b2 = (BUILD / "bank2.bin").read_bytes()
    lo5, hi5 = syms["ORG_START"], syms["BANK5_END"]
    out.append((0, lo5, b5[lo5 - 0x4000:hi5 - 0x4000]))
    lo2, hi2 = syms["EXP63"], syms["HOT_END"]
    out.append((0, lo2, b2[lo2 - 0x8000:hi2 - 0x8000]))
    for n in (0, 1, 3, 4, 6, 7):
        data = (BUILD / f"bank{n}.bin").read_bytes().rstrip(b"\0")
        out.append((n, 0xC000, data))
    return out


def build_loader(blocks, start: int) -> bytes:
    src = BUILD / "loader.asm"
    lines = [f"        org {LOADER_ORG}",
             "loader: di",
             "        ld hl,blocks",
             ".next:  ld a,(hl)",
             "        cp 0xFF",
             "        jr z,.go",
             "        or 0x10",
             "        ld bc,0x7FFD",
             "        out (c),a",
             "        inc hl",
             "        ld e,(hl)",
             "        inc hl",
             "        ld d,(hl)",
             "        inc hl",
             "        push de",
             "        pop ix",
             "        ld e,(hl)",
             "        inc hl",
             "        ld d,(hl)",
             "        inc hl",
             "        push hl",
             "        ld a,0xFF",
             "        scf",
             "        call 0x0556                 ; LD-BYTES",
             "        di",
             "        pop hl",
             "        jr c,.next",
             "        rst 0                       ; ошибка ленты — сброс",
             ".go:    ld a,0x10",
             "        ld bc,0x7FFD",
             "        out (c),a",
             f"        jp {start}",
             "blocks:"]
    for bank, addr, data in blocks:
        lines.append(f"        db {bank}")
        lines.append(f"        dw {addr}, {len(data)}")
    lines.append("        db 0xFF")
    src.write_text("\n".join(lines) + "\n", encoding="utf-8")
    raw = BUILD / "loader.bin"
    r = subprocess.run([str(SJASM), "--nologo", f"--raw={raw}", src.as_posix()], capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(r.stdout + r.stderr)
    return raw.read_bytes()


def main():
    syms = load_syms()
    blocks = chunks(syms)
    loader = build_loader(blocks, syms["start"])
    scr = (ROOT / "art" / "out" / "hippogpt.scr").read_bytes()
    bas = basic_loader()
    tap = bytearray()
    tap += header(0, "HippoGPT", len(bas), 10, len(bas))
    tap += tap_block(0xFF, bas)
    tap += header(3, "hippo ldr", len(loader), LOADER_ORG, 32768)
    tap += tap_block(0xFF, loader)
    tap += header(3, "hippo scr", len(scr), 16384, 32768)
    tap += tap_block(0xFF, scr)
    total = len(scr) + len(loader)
    for bank, addr, data in blocks:
        tap += tap_block(0xFF, data)
        total += len(data)
        print(f"  банк {bank} {addr:#06x}: {len(data)} байт")
    out = BUILD / "hippo.tap"
    out.write_bytes(bytes(tap))
    print(f"{out}: {len(tap)} байт, данных {total}, при 1500 бод ~{total * 8 / 1500 / 60:.0f} мин")


if __name__ == "__main__":
    main()
