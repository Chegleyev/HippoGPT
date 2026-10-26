"""Дискета HippoGPT 2 в 1: build/hippo.trd и build/hippo.scl — версии 128K (build) и 256 (build256) на одном диске.

Загрузчик (#6000) сам выбирает версию:
  * если страница 8 отдельна от страницы 0 по одной из схем — версия 256:
        бит 6 порта #7FFD — Pentagon 512/1024, ZX Evolution, MiSTer;
        бит 4 порта #1FFD — Scorpion ZS-256, KAY;
        бит 0 порта #DFFD — Profi 1024.
    Порядок проверки такой из-за неполной дешифрации: на Пентагоне запись в #1FFD попадает в #7FFD, а #DFFD у
    машин без этого порта — выбор регистра AY (безвредно). Найденная схема ставится в переход setbank версии 256
    (jp sb_pent / sb_scorp / sb_profi, z80/hippo.asm);
  * иначе, если страница 1 отдельна от страницы 0, — версия 128K;
  * иначе сообщение, что нужно 128K, и выход в Бейсик.
Файлы ищутся по каталогу (их можно копировать на другой диск по одному) и читаются функцией 5 TR-DOS (#3D13)
прямо в нужную страницу. Шапка упакована ZX0 (tools/zx0enc.py, распаковщик Эйнара Сауказа, third_party/zx0) и
проявляется сверху вниз по строкам атрибутов.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mktrd import CAT, LOADER_ORG, SECTORS, TRACKS, SIDES, Disk, basic_boot, padded  # noqa: E402
from zx import SJASM  # noqa: E402
from zx128 import ROOT, load_syms  # noqa: E402
from zx0enc import compress  # noqa: E402

B128 = ROOT / "build"
B256 = ROOT / "build256"
DZX0 = ROOT / "third_party" / "zx0" / "dzx0_standard.asm"
SCRBUF = 0x8800                 # упакованная шапка (после каталога 8000-87FF)
DEPACK = 0x9000                 # распакованная шапка; шапка грузится первой, до hippo_hi (A200..)
SCR_PAGE = 0xFF                 # метка в таблице файлов: это шапка — распаковать и показать


def plan(build: Path, tag: str, pages):
    """[(имя, страница, адрес, образ, начало в образе, длина)] — что грузить для одной версии."""
    syms = load_syms(build / "hippo.sym")
    assert syms["ORG_START"] >= 0x6700 and syms["EXP63"] >= CAT + 0x800      # каталог нужен до конца загрузки
    b5, b2 = (build / "bank5.bin").read_bytes(), (build / "bank2.bin").read_bytes()
    out = [(f"h{tag}lo", 0, syms["ORG_START"], b5, syms["ORG_START"] - 0x4000, syms["BANK5_END"] - syms["ORG_START"]),
           (f"h{tag}hi", 0, syms["EXP63"], b2, syms["EXP63"] - 0x8000, syms["HOT_END"] - syms["EXP63"])]
    for n in pages:
        f = build / f"bank{n}.bin"
        img = f.read_bytes() if f.exists() else b""
        if img.rstrip(b"\0"):
            out.append((f"h{tag}p{n:X}", n, 0xC000, img, 0, len(img.rstrip(b"\0"))))
    return syms, out


def file_table(label, files):
    lines = [f"{label}:"]
    for name, page, addr, *_ in files:
        lines += [f'        db "{name:<8}","C",{page}', f"        dw {addr}"]
    return lines + ["        db 0"]


def build_loader(scr_file, f128, f256, s128, s256) -> bytes:
    L = [f"        org {LOADER_ORG}",
         "loader: ld a,0x10                   ; страница 0, ПЗУ 48K",
         "        ld bc,0x7FFD",
         "        out (c),a",
         f"        ld hl,{CAT}",
         "        ld de,0                     ; дорожка 0, сектор 0",
         "        ld bc,0x0805                ; 8 секторов каталога, функция 5 — чтение",
         "        call 0x3D13",
         "        ld ix,f_scr",
         "        call load",
         "        ld hl,page_pent             ; какая схема даёт страницу 8",
         f"        ld de,{s256['sb_pent']}",
         "        call try8",
         "        jr z,.big",
         "        ld hl,page_scorp",
         f"        ld de,{s256['sb_scorp']}",
         "        call try8",
         "        jr z,.big",
         "        ld hl,page_profi",
         f"        ld de,{s256['sb_profi']}",
         "        call try8",
         "        jr z,.big",
         "        ld hl,page_128",
         "        ld a,1",
         "        call try",
         "        jr nz,.small",
         "        ld ix,f_128",
         "        call load",
         "        di",
         "        xor a",
         "        call page",
         f"        jp {s128['start']}",
         ".big:   push de",
         "        ld ix,f_256",
         "        call load",
         "        pop hl",
         "        di",
         f"        ld ({s256['setbank'] + 1}),hl     ; setbank версии 256: jp sb_...",
         "        xor a",
         "        call page",
         f"        jp {s256['start']}",
         ".small: ld a,2                      ; меньше 128K: сообщение и назад в Бейсик",
         "        call 0x1601",
         "        ld hl,msg",
         ".pr:    ld a,(hl)",
         "        or a",
         "        ret z",
         "        rst 0x10",
         "        inc hl",
         "        jr .pr",
         "",
         "; try8 / try: HL = процедура страницы, A = проверяемая страница (try8 — 8). ZF = 1, если она не совпала",
         "; со страницей 0 (метки в обе, читаем страницу 0). Процедура остаётся в page. DE сохраняет.",
         "try8:   ld a,8",
         "try:    ld (page + 1),hl",
         "        push de",
         "        push af",
         "        xor a",
         "        call page",
         "        ld a,0x55",
         "        ld (0xC000),a",
         "        pop af",
         "        call page",
         "        ld a,0xAA",
         "        ld (0xC000),a",
         "        xor a",
         "        call page",
         "        ld a,(0xC000)",
         "        pop de",
         "        cp 0x55",
         "        ret",
         "",
         "; page: A = страница на C000 (ПЗУ 48K), портит BC. Переход ставит try.",
         "page:   jp page_128",
         "page_pent:                          ; бит 3 номера -> бит 6 #7FFD",
         "        bit 3,a",
         "        jr z,page_128",
         "        xor 0x48",
         "page_128:",
         "        or 0x10",
         "        ld bc,0x7FFD",
         "        out (c),a",
         "        ret",
         "page_scorp:                         ; бит 3 номера -> бит 4 #1FFD",
         "        push af",
         "        and 8",
         "        add a,a",
         "        ld bc,0x1FFD",
         "        out (c),a",
         "        pop af",
         "        and 7",
         "        jr page_128",
         "page_profi:                         ; бит 3 номера -> бит 0 #DFFD",
         "        push af",
         "        and 8",
         "        rrca",
         "        rrca",
         "        rrca",
         "        ld bc,0xDFFD",
         "        out (c),a",
         "        pop af",
         "        and 7",
         "        jr page_128",
         "",
         "; load: IX = таблица файлов (имя 8, тип, страница, адрес; 0 — конец) -> каждый файл по каталогу в память",
         "load:   ld a,(ix+0)",
         "        or a",
         "        ret z",
         f"        ld hl,{CAT}",
         ".find:  ld a,(hl)",
         "        or a",
         "        jr z,.miss                  ; конец каталога",
         "        push hl",
         "        push ix",
         "        pop de",
         "        ld b,9                      ; имя + тип",
         ".cmp:   ld a,(de)",
         "        cp (hl)",
         "        jr nz,.ne",
         "        inc hl",
         "        inc de",
         "        djnz .cmp",
         "        pop hl                      ; нашли: +13 секторов, +14 сектор, +15 дорожка",
         "        ld bc,13",
         "        add hl,bc",
         "        ld b,(hl)",
         "        inc hl",
         "        ld e,(hl)",
         "        inc hl",
         "        ld d,(hl)",
         "        ld a,(ix+9)                 ; страница на C000",
         f"        cp {SCR_PAGE}",
         "        jr z,.np",
         "        push bc",
         "        call page",
         "        pop bc",
         ".np:    ld l,(ix+10)",
         "        ld h,(ix+11)",
         "        ld c,5",
         "        push ix",
         "        call 0x3D13",
         "        pop ix",
         "        ld a,(ix+9)",
         f"        cp {SCR_PAGE}",
         "        call z,show",
         "        ld de,12",
         "        add ix,de",
         "        jr load",
         ".ne:    pop hl",
         "        ld bc,16",
         "        add hl,bc",
         "        ld a,h",
         f"        cp {(CAT + 0x800) >> 8}",
         "        jr nz,.find",
         ".miss:  ld a,2                      ; файла нет — красная рамка",
         "        out (0xFE),a",
         "        di",
         "        halt",
         "",
         "; show: шапка из SCRBUF (ZX0) -> экран: атрибуты чёрные, пиксели, потом атрибуты по строке сверху вниз",
         "show:   push ix",
         "        ld hl,0x5800",
         "        ld de,0x5801",
         "        ld bc,767",
         "        ld (hl),0",
         "        ldir",
         f"        ld hl,{SCRBUF}",
         f"        ld de,{DEPACK}",
         "        call dzx0_standard",
         f"        ld hl,{DEPACK}",
         "        ld de,0x4000",
         "        ld bc,6144",
         "        ldir",
         "        ld de,0x5800",
         ".row:   ld bc,32",
         "        ldir",
         "        ld bc,1200                  ; ~1/100 с на строку",
         ".w:     dec bc",
         "        ld a,b",
         "        or c",
         "        jr nz,.w",
         "        ld a,d                      ; строк осталось: (0x5B00 - DE) / 32",
         "        cp 0x5B",
         "        jr nz,.row",
         "        pop ix",
         "        ret",
         "",
         f'        include "{DZX0.as_posix()}"',
         "",
         'msg:    db "HippoGPT needs 128K memory.", 13, 0']
    L += file_table("f_scr", [scr_file]) + file_table("f_128", f128) + file_table("f_256", f256)
    src = B128 / "loader_disk.asm"
    src.write_text("\n".join(L) + "\n", encoding="utf-8")
    raw = B128 / "loader_disk.bin"
    r = subprocess.run([str(SJASM), "--nologo", f"--raw={raw}", src.as_posix()], capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(r.stdout + r.stderr)
    data = raw.read_bytes()
    assert LOADER_ORG + len(data) <= 0x6700
    return data


def scl(files) -> bytes:
    """files: [(имя, тип, p1, p2, данные по секторам)] -> образ SCL."""
    out = bytearray(b"SINCLAIR" + bytes([len(files)]))
    for name, kind, p1, p2, data in files:
        out += name.encode("ascii").ljust(8, b" ")[:8] + kind.encode() + p1.to_bytes(2, "little") \
            + p2.to_bytes(2, "little") + bytes([len(data) // 256])
    for *_, data in files:
        out += data
    return bytes(out + (sum(out) & 0xFFFFFFFF).to_bytes(4, "little"))


def main():
    s128, f128 = plan(B128, "128", (0, 1, 3, 4, 6, 7))
    s256, f256 = plan(B256, "256", [n for n in range(16) if n not in (2, 5)])
    scr = (ROOT / "art" / "out" / "hippogpt.scr").read_bytes()
    packed = compress(scr)
    assert SCRBUF + len(packed) <= DEPACK
    scr_file = ("hipposcr", SCR_PAGE, SCRBUF, packed, 0, len(packed))
    loader = build_loader(scr_file, f128, f256, s128, s256)
    boot = basic_boot()
    files = [("boot", "B", len(boot), len(boot), boot + bytes([0x80, 0xAA, 10, 0])),
             ("hippo", "C", LOADER_ORG, len(loader), loader)]
    for nm, pg, addr, img, lo, n in [scr_file] + f128 + f256:
        assert addr % 256 == 0
        files.append((nm, "C", addr, n, padded(img, lo, n)))
    d = Disk("HippoGPT")
    for nm, kind, p1, p2, data in files:
        d.add(nm, kind, p1, p2, data)
    (B128 / "hippo.trd").write_bytes(d.finish())
    sec = [(nm, kind, p1, p2, data + bytes(-len(data) % 256)) for nm, kind, p1, p2, data in files]
    (B128 / "hippo.scl").write_bytes(scl(sec))
    for nm, kind, p1, p2, data in files:
        print(f"  {nm:<8} {kind} {p1:#06x} {p2:6d}  {len(data) // 256 + (len(data) % 256 > 0):3d} сект.")
    print(f"{B128 / 'hippo.trd'}, hippo.scl: {d.files} файлов, {d.free - SECTORS} секторов из "
          f"{TRACKS * SIDES * SECTORS - SECTORS}; загрузчик {len(loader)} байт, шапка {len(scr)} -> {len(packed)}")


if __name__ == "__main__":
    main()
