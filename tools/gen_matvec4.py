"""Ядро умножения матрицы на вектор с 4-битными весами (z80/gen_matvec4.asm).

Вес — код c в -7..7. Пока матрица считается, вход x не меняется, поэтому для каждого входа
заранее строится таблица кратных c*x (pt_build), а вес хранится байтом, который и есть смещение
нужного кратного в таблице: смещение = (i & 7) * 32 + (c + 7) * 2. Блок входа i — 32 байта по
адресу PT + 32*i (8 входов на страницу), страницу ядро меняет INC H через каждые 8 весов.

На вес: ld a,(de) / inc e / ld l,a / ld a,(hl) / add a,c / ld c,a / inc l / ld a,(hl) / adc a,b /
ld b,a — 49 тактов. Аккумулятор 16-битный: сумма должна помещаться в int16 (проверяется на эталоне).

Строки весов лежат подряд и выровнены по своей длине (64 или 256 байт): внутри строки хватает
INC E, после строки — перенос в D.
"""
from pathlib import Path

import numpy as np

BLOCK = 64          # весов в развёрнутом теле цикла


def gen(brk: str | None = None) -> str:
    """brk — метка, куда прыгать по BREAK/SPACE (проверка в начале строки, 32 такта); None — без проверки."""
    out = []
    o = out.append
    o("; сгенерировано tools/gen_matvec4.py — не править руками")
    o("; pt_build: B = число входов (0 = 256), IX -> входы (слова) -> таблица кратных в PT")
    o("pt_build:")
    o("        ld (pt_sp+1),sp")
    o("        ld hl,PT + 30")
    o("        ld (pt_dst+1),hl")
    o("        ld a,b                  ; счётчик входов — в A' (B занят под -x)")
    o("        ex af,af'")
    o("pt_in:  ld e,(ix+0)")
    o("        ld d,(ix+1)")
    o("        inc ix")
    o("        inc ix")
    o("        xor a                   ; BC = -x")
    o("        sub e")
    o("        ld c,a")
    o("        sbc a,a")
    o("        sub d")
    o("        ld b,a")
    o("        ld h,d                  ; HL = 7x")
    o("        ld l,e")
    o("        add hl,hl")
    o("        add hl,hl")
    o("        add hl,hl")
    o("        add hl,bc")
    o("pt_dst: ld sp,0")
    for _ in range(15):
        o("        push hl")
        o("        add hl,bc")
    o("        ld hl,(pt_dst+1)")
    o("        ld de,32")
    o("        add hl,de")
    o("        ld (pt_dst+1),hl")
    o("        ex af,af'")
    o("        dec a")
    o("        jr z,pt_end")
    o("        ex af,af'")
    o("        jp pt_in")
    o("pt_end:")
    o("pt_sp:  ld sp,0")
    o("        ret")
    o("")
    o("; matvec4: DE -> веса, IX -> выход (слова), IYH = строк (0 = 256), mv4_n+2 = длина строки / 64")
    o("matvec4:")
    o("mv4_row:")
    if brk:                                       # BREAK/SPACE в начале каждой строки (A здесь свободен)
        o("        ld a,0x7F")
        o("        in a,(0xFE)")
        o("        rra")
        o(f"        jp nc,{brk}")
    o("        ld h,PT >> 8")
    o("        ld bc,0")
    o("mv4_n:  ld iyl,0")
    o("mv4_blk:")
    for g in range(BLOCK // 8):
        for _ in range(8):
            o("        ld a,(de)")
            o("        inc e")
            o("        ld l,a")
            o("        ld a,(hl)")
            o("        add a,c")
            o("        ld c,a")
            o("        inc l")
            o("        ld a,(hl)")
            o("        adc a,b")
            o("        ld b,a")
        o("        inc h")
    o("        dec iyl")
    o("        jp nz,mv4_blk")
    o("        ld a,e                  ; строка кончилась на границе страницы — перенос")
    o("        or a")
    o("        jr nz,mv4_nc")
    o("        inc d")
    o("mv4_nc: ld (ix+0),c")
    o("        ld (ix+1),b")
    o("        inc ix")
    o("        inc ix")
    o("        dec iyh")
    o("        jp nz,mv4_row")
    o("        ret")
    return "\n".join(out) + "\n"


def pack_rows(codes) -> bytes:
    """codes: строки кодов -7..7 -> байты-смещения в таблице кратных."""
    codes = np.asarray(codes, dtype=np.int64)
    assert codes.min() >= -7 and codes.max() <= 7
    i = np.arange(codes.shape[1])
    return ((i[None, :] & 7) * 32 + (codes + 7) * 2).astype(np.uint8).tobytes()


if __name__ == "__main__":
    dst = Path(__file__).resolve().parent.parent / "z80" / "gen_matvec4.asm"
    dst.write_text(gen(), encoding="utf-8")
    print(dst)
