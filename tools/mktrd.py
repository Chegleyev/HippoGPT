"""Общие части дискеты TR-DOS (80 дорожек, 2 стороны): Бейсик boot, образ диска, хвосты секторов.

Сама дискета (версии 128K и 256 на одном диске, загрузчик выбирает по машине) — tools/mkdisk.py.
    boot     B  10 CLEAR 24575: RANDOMIZE USR 15619: REM : LOAD "hippo" CODE
                20 RANDOMIZE USR 24576
Хвост последнего сектора каждого файла — те же байты образа, поэтому дочитывание целого сектора безопасно.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mktap import bas_num  # noqa: E402

LOADER_ORG = 0x6000
CAT = 0x8000                    # каталог при загрузке (там потом таблицы PT — их строит программа)
TRACKS, SIDES, SECTORS = 80, 2, 16


def basic_boot() -> bytes:
    CLEAR, RAND, USR, REM, LOAD, CODE = 0xFD, 0xF9, 0xC0, 0xEA, 0xEF, 0xAF
    l10 = (bytes([CLEAR]) + bas_num(LOADER_ORG - 1) + b":" + bytes([RAND, USR]) + bas_num(15619) + b":"
           + bytes([REM]) + b":" + bytes([LOAD]) + b'"hippo"' + bytes([CODE]) + b"\r")
    l20 = bytes([RAND, USR]) + bas_num(LOADER_ORG) + b"\r"
    out = b""
    for num, line in [(10, l10), (20, l20)]:
        out += num.to_bytes(2, "big") + len(line).to_bytes(2, "little") + line
    return out


class Disk:
    def __init__(self, label: str):
        self.img = bytearray(TRACKS * SIDES * SECTORS * 256)
        self.files = 0
        self.free = 1 * SECTORS             # первый свободный сектор: дорожка 1, сектор 0 (линейно)
        self.label = label

    def add(self, name: str, kind: str, p1: int, p2: int, data: bytes):
        """p1, p2 — поля 9-10 и 11-12 записи каталога. data уже со всем хвостом."""
        n = (len(data) + 255) // 256
        assert n <= 255 and self.files < 128
        pos = self.free
        self.img[pos * 256:pos * 256 + len(data)] = data
        e = name.encode("ascii").ljust(8, b" ")[:8] + kind.encode() + p1.to_bytes(2, "little") \
            + p2.to_bytes(2, "little") + bytes([n, pos % SECTORS, pos // SECTORS])
        self.img[self.files * 16:self.files * 16 + 16] = e
        self.files += 1
        self.free += n

    def finish(self) -> bytes:
        s = 8 * 256                          # служебный сектор: дорожка 0, сектор 8
        total = TRACKS * SIDES * SECTORS
        self.img[s + 0xE1] = self.free % SECTORS
        self.img[s + 0xE2] = self.free // SECTORS
        self.img[s + 0xE3] = 0x16            # 80 дорожек, 2 стороны
        self.img[s + 0xE4] = self.files
        self.img[s + 0xE5:s + 0xE7] = (total - self.free).to_bytes(2, "little")
        self.img[s + 0xE7] = 0x10            # признак TR-DOS
        self.img[s + 0xE9:s + 0xF2] = b" " * 9
        self.img[s + 0xF5:s + 0xFD] = self.label.encode("ascii").ljust(8, b" ")[:8]
        return bytes(self.img)


def padded(img: bytes, lo: int, n: int) -> bytes:
    """n байт образа с lo плюс хвост образа до границы сектора."""
    m = (n + 255) // 256 * 256
    out = img[lo:lo + m]
    return out + bytes(m - len(out))
