"""Дискета HippoGPT (build/hippo.trd, tools/mkdisk.py) в Unreal Speccy 0.39.0 на разных машинах.

    python tools/unreal_test.py all [затравка ...]           все машины из MACHINES
    python tools/unreal_test.py scorpion [затравка ...]      одна машина
    python tools/unreal_test.py quit

Для каждой машины: unreal.ini (модель памяти, её ПЗУ, сброс в TR-DOS, диск), запуск, ожидание загрузки, какая
версия загрузилась (128K или 256) и какой переход стоит в setbank, страницы против сборки (страницы 8..15 ищутся
среди всех физических: у Пентагона в Unreal бит 6 #7FFD — это бит 4 номера), затравки и сверка сказок с
эталоном, экран -> build/unreal_<машина>.png.

Клавиатура и экран — через память процесса unreal.exe (ReadProcessMemory/WriteProcessMemory), окно и фокус не
нужны: ОЗУ Спектрума у Unreal — массив memory[], страница n по смещению n*#4000 (emul.h). Нажатие клавиши: в
KEYMAP на месте «q» ставится нужный символ, а операнд «ld a,255» в scan_key («ничего не нажато») меняется на код
клавиши «q»; через 0,15 с байт возвращается, getkey видит отпускание. Так идут input_line, эхо и разбор затравки,
как при наборе руками; не проверяется только опрос портов клавиатуры.

Сверка: из памяти берутся затравка (PROMPT), температура (TQ), токены (GENLOG в банке 7, GEN_N) и состояние
генератора RND после сказки. Затравка генератора (её дал регистр R в момент ENTER) восстанавливается обратным
ходом xorshift16 на число выборок, и сказка сравнивается с ref/intref.py (tools/test_story128.py).
В unreal.ini выключен звук ([SOUND] Enabled=0 — максимальная скорость) и стоит driver=gdi (blt падает).
"""
from __future__ import annotations

import ctypes
import ctypes.wintypes as W
import json
import os
import pickle
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ref"))
import sampler  # noqa: E402
from test_story128 import reference_story, screen_png  # noqa: E402
from zx128 import ROOT, load_syms  # noqa: E402

EMU = Path(os.environ.get("UNREAL_DIR", ROOT / ".tools" / "unreal039"))   # unreal.exe, unreal.ini, ПЗУ
BUILDS = {"128K": ROOT / "build", "256": ROOT / "build256"}
PAGE = 0x4000
KEY_Q = 10                                   # код клавиши «q» у scan_key: полуряд 2, бит 0
# модель памяти Unreal, размер, тайминги, набор ПЗУ ([ROM.*] в unreal.ini) -> какая версия должна загрузиться
MACHINES = {
    "pentagon128":  ("PENTAGON", 128, "PENTAGON", "pentagon", "128K"),
    "pentagon512":  ("PENTAGON", 512, "PENTAGON", "pentagon", "256"),
    "pentagon1024": ("PENTAGON", 1024, "PENTAGON", "pentagon", "256"),
    "scorpion":     ("SCORPION", 256, "SCORPION", "scorpion", "256"),
    "scorpion1024": ("SCORPION", 1024, "SCORPION", "scorpion", "256"),
    "kay1024":      ("KAY", 1024, "KAY1024", "KAY1", "256"),
    "profi":        ("PROFI", 1024, "PROFI", "profi", "256"),
}
# ZX Evolution (HIMEM=ATM3, 4096, ROMSET пустой — zxevo.rom) в Unreal до TR-DOS не доходит: BIOS ждёт настройки.
PROMPTS = ["Dizzy the egg", "The dragon was hungry"]
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)


class MBI(ctypes.Structure):
    _fields_ = [("BaseAddress", ctypes.c_void_p), ("AllocationBase", ctypes.c_void_p),
                ("AllocationProtect", W.DWORD), ("PartitionId", W.WORD), ("RegionSize", ctypes.c_size_t),
                ("State", W.DWORD), ("Protect", W.DWORD), ("Type", W.DWORD)]


def unreal_pid():
    out = subprocess.run(["tasklist", "/FI", "IMAGENAME eq unreal.exe", "/FO", "CSV", "/NH"],
                         capture_output=True, text=True).stdout
    pids = [int(line.split('","')[1]) for line in out.splitlines() if line.startswith('"unreal.exe"')]
    return pids[0] if pids else None


class Proc:
    """Память процесса unreal.exe."""

    def __init__(self):
        self.h = kernel32.OpenProcess(0x0010 | 0x0020 | 0x0008 | 0x0400, False, unreal_pid())
        if not self.h:
            raise OSError(ctypes.get_last_error(), "OpenProcess")

    def read(self, addr, n):
        buf = ctypes.create_string_buffer(n)
        got = ctypes.c_size_t()
        if not kernel32.ReadProcessMemory(self.h, ctypes.c_void_p(addr), buf, n, ctypes.byref(got)):
            return None
        return buf.raw[:got.value]

    def write(self, addr, data: bytes):
        if not kernel32.WriteProcessMemory(self.h, ctypes.c_void_p(addr), data, len(data), None):
            raise OSError(ctypes.get_last_error(), "WriteProcessMemory")

    def find_base(self, sig: bytes, off: int, ok):
        """База memory[]: подпись sig лежит по base + off, и ok(base) подтверждает (подпись есть и в образе диска)."""
        addr, mbi = 0, MBI()
        while kernel32.VirtualQueryEx(self.h, ctypes.c_void_p(addr), ctypes.byref(mbi), ctypes.sizeof(mbi)):
            if mbi.State == 0x1000 and mbi.Protect in (0x04, 0x08, 0x40):
                data = self.read(addr, mbi.RegionSize) or b""
                i = data.find(sig)
                while i >= 0:
                    if ok(addr + i - off):
                        return addr + i - off
                    i = data.find(sig, i + 1)
            addr += mbi.RegionSize
            if addr >= 1 << 47:
                break
        return None


class Emu:
    """Хиппо одной из версий в ОЗУ Спектрума внутри unreal.exe."""

    def __init__(self, proc: Proc, version: str, base: int):
        self.p, self.version, self.base = proc, version, base
        self.build = BUILDS[version]
        self.S = load_syms(self.build / "hippo.sym")

    @staticmethod
    def attach(proc: Proc):
        """Какая версия загружена: подписи кода по main (страница 5) и forward (страница 2)."""
        for version, build in BUILDS.items():
            S = load_syms(build / "hippo.sym")
            b5, b2 = (build / "bank5.bin").read_bytes(), (build / "bank2.bin").read_bytes()
            o5, o2 = S["main"] - 0x4000, S["forward"] - 0x8000
            sig2 = b2[o2:o2 + 64]
            base = proc.find_base(b5[o5:o5 + 64], 5 * PAGE + o5,
                                  lambda b: proc.read(b + 2 * PAGE + o2, 64) == sig2)
            if base is not None:
                return Emu(proc, version, base)
        return None

    def page(self, n, off=0, size=PAGE):
        return self.p.read(self.base + n * PAGE + off, size)

    def mem(self, addr, size):
        """4000-BFFF: страницы 5 и 2 — там у Хиппо переменные."""
        assert 0x4000 <= addr and addr + size <= 0xC000
        return self.page(5 if addr < 0x8000 else 2, addr & 0x3FFF, size)

    def word(self, addr):
        return int.from_bytes(self.mem(addr, 2), "little")

    def poke(self, addr, data: bytes):
        assert 0x4000 <= addr and addr + len(data) <= 0xC000
        self.p.write(self.base + (5 if addr < 0x8000 else 2) * PAGE + (addr & 0x3FFF), data)

    def scheme(self):
        """Куда ведёт переход setbank (версия 256): его ставит загрузчик."""
        if "sb_scorp" not in self.S:
            return "#7FFD"
        target = self.word(self.S["setbank"] + 1)
        return {self.S["sb_pent"]: "бит 6 #7FFD", self.S["sb_scorp"]: "бит 4 #1FFD",
                self.S["sb_profi"]: "бит 0 #DFFD"}.get(target, f"{target:#06x}?")

    def type(self, text: str):
        """Набор через getkey (см. выше)."""
        b5 = (self.build / "bank5.bin").read_bytes()
        o = self.S["scan_key"] - 0x4000
        imm = self.S["scan_key"] + b5.find(bytes([0x3E, 0xFF, 0xC9]), o, o + 80) - o + 1   # операнд «ld a,255»
        q = self.S["KEYMAP"] + KEY_Q
        assert self.mem(imm, 1) == b"\xff" and self.mem(q, 1) == b"q"
        try:
            for ch in text:
                self.poke(q, ch.encode("ascii"))
                self.poke(imm, bytes([KEY_Q]))
                time.sleep(0.15)
                self.poke(imm, b"\xff")
                time.sleep(0.15)
        finally:
            self.poke(q, b"q")
            self.poke(imm, b"\xff")

    def screen(self, name):
        out = ROOT / "build" / f"{name}.png"
        screen_png(self.page(5, 0, 6912), out)
        return out


def check_pages(emu: Emu, ram_pages: int):
    """Что загрузил загрузчик — против образов страниц сборки. -> (всё ли совпало, {логическая: физическая})."""
    S, ok, where = emu.S, True, {}
    for n in range(16):
        f = emu.build / f"bank{n}.bin"
        if not f.exists():
            continue
        img = f.read_bytes()
        if n == 5:
            lo, hi = S["main"] - 0x4000, S["TXT_ATTR"] - 0x4000                   # код; вокруг — буферы и переменные
        elif n == 2:
            lo, hi = S["forward"] - 0x8000, S["HOT_END"] - 0x8000                # setbank с переходом — раньше
        else:
            lo, hi = 0, len(img.rstrip(b"\0"))
        if hi <= lo:
            continue
        want = img[lo:hi]
        cands = [n] + [k for k in range(ram_pages) if k != n] if n >= 8 else [n]
        phys = next((k for k in cands if emu.page(k, lo, hi - lo) == want), None)
        if phys is None and n == 7:                                             # кэш внимания уже писался
            phys = 7 if emu.page(7, 0, 0x1000) == want[:0x1000] else None
        where[n] = phys
        ok &= phys is not None
    return ok, where


def check_story(emu: Emu, typed: str):
    S = emu.S
    n = emu.word(S["GEN_N"])
    log = emu.page(7, S["GENLOG"] - 0xC000, 2 * n)
    got = [int.from_bytes(log[i:i + 2], "little") for i in range(0, 2 * n, 2)]
    # PROMPT_LEN после сказки уже обнулён новым input_line, а сама строка в PROMPT осталась
    typed = typed[:S["PROMPT_MAX"]]                              # лишнее input_line не берёт
    prompt = emu.mem(S["PROMPT"], len(typed)).decode("ascii")
    assert prompt == typed, (prompt, typed)
    tq, rnd = emu.word(S["TQ"]), emu.word(S["RND"])
    with open(emu.build / ("intmodel256.pkl" if emu.version == "256" else "intmodel128.pkl"), "rb") as f:
        im = pickle.load(f)
    meta = json.loads((ROOT / "data" / f"{im.data}.json").read_text(encoding="utf-8"))
    pieces, merges = meta["pieces"], [tuple(x) for x in meta["merges"]]
    lm, _ = sampler.logit_mults(im.logit_row)
    heal = S.get("HEAL_WORDS", 1)
    last = sampler.tokenize_bpe(prompt, merges)[-1:]
    head = prompt                                                # склейка: первый токен уже содержит хвост затравки
    if last and (heal or not prompt[-1].isalpha()):
        head = prompt[:len(prompt) - len(pieces[last[0]])]
    text = head + "".join(pieces[t] for t in got)
    for samples in (n + 1, n):                                   # с выбранным концом сказки и без него
        seed = rnd
        for _ in range(samples):
            seed = unshift(seed)
        ref = reference_story(im, pieces, merges, lm, tq, prompt, seed, heal, S.get("MIN_EOS", 0))
        if ref == got:
            return True, n, text
    return False, n, text


def unshift(x: int) -> int:
    """Обратный ход sampler.xorshift16."""
    x ^= (x << 8) & 0xFFFF
    x ^= x >> 9
    x ^= ((x << 7) ^ (x << 14)) & 0xFFFF
    return x & 0xFFFF


def story(emu: Emu, prompt: str, timeout=1800):
    emu.type(prompt)
    emu.poke(emu.S["GEN_N"], b"\0\0")
    t0 = time.time()
    emu.type("\r")
    last, still = -1, 0
    while time.time() - t0 < timeout:
        time.sleep(1)
        n = emu.word(emu.S["GEN_N"])
        still = still + 1 if n == last and n > 0 else 0
        if still >= 2:
            break
        last = n
    return check_story(emu, prompt)


def quit_emu():
    subprocess.run(["taskkill", "/F", "/IM", "unreal.exe"], capture_output=True)   # WM_CLOSE без рабочего стола не доходит
    time.sleep(1)


def configure(himem, ram, preset, romset):
    ini = EMU / "unreal.ini"
    text = ini.read_bytes().decode("cp1251")
    for key, val in (("HIMEM", himem), ("RAMSize", str(ram)), ("Preset", preset), ("ROMSET", romset),
                     ("RESET", "DOS"), ("diskA", "hippo.trd")):
        text, n = re.subn(rf"(?m)^({key}=)[^;\r\n]*", lambda m: m.group(1) + val + " ", text, count=1)
        assert n == 1, key
    ini.write_bytes(text.encode("cp1251"))
    (EMU / "hippo.trd").write_bytes((ROOT / "build" / "hippo.trd").read_bytes())


def loader_screen(proc: Proc, name):
    """Хиппо не нашёлся: экран по загрузчику (#6000 в странице 5, системная переменная CHARS = #3C00)."""
    sig = (ROOT / "build" / "loader_disk.bin").read_bytes()[:64]
    base = proc.find_base(sig, 5 * PAGE + 0x2000, lambda b: proc.read(b + 5 * PAGE + 0x1C36, 2) == b"\x00\x3c")
    if base is None:
        return None
    out = ROOT / "build" / f"{name}.png"
    screen_png(proc.read(base + 5 * PAGE, 6912), out)
    return out


def run(machine: str, prompts):
    himem, ram, preset, romset, expect = MACHINES[machine]
    quit_emu()
    configure(himem, ram, preset, romset)
    subprocess.Popen([str(EMU / "unreal.exe")], cwd=EMU)
    t0, emu = time.time(), None
    while time.time() - t0 < 40 and emu is None:
        time.sleep(3)
        if unreal_pid() is None:
            print(f"{machine}: эмулятор закрылся")
            return False
        emu = Emu.attach(Proc())
    if emu is None:
        png = loader_screen(Proc(), f"unreal_{machine}")
        print(f"{machine}: Хиппо не загрузился за 40 с; экран: {png or 'загрузчик тоже не найден'}")
        quit_emu()
        return False
    time.sleep(1)
    ok_pages, where = check_pages(emu, ram // 16)
    moved = {n: k for n, k in where.items() if k != n}
    print(f"{machine}: версия {emu.version} ({emu.scheme()}), страницы {'OK' if ok_pages else 'РАСХОДЯТСЯ'}"
          + (f", логические -> физические {moved}" if moved else ""))
    ok = ok_pages and emu.version == expect
    for k, p in enumerate(prompts, 1):
        same, n, text = story(emu, p)
        ok &= same
        print(f"  сказка {k}: {'OK' if same else 'РАСХОДИТСЯ'}, {n} токенов: {text[:90]!r}")
        if k == 1:
            print("  экран:", emu.screen(f"unreal_{machine}"))
    quit_emu()
    return ok


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd == "quit":
        quit_emu()
        return
    machines = list(MACHINES) if cmd == "all" else [cmd]
    if any(m not in MACHINES for m in machines):
        sys.exit(__doc__)
    prompts = sys.argv[2:] or PROMPTS
    res = {m: run(m, prompts) for m in machines}
    configure(*MACHINES["scorpion"][:4])                         # по умолчанию — Scorpion
    print("итог:", ", ".join(f"{m} {'OK' if r else 'НЕТ'}" for m, r in res.items()))
    sys.exit(0 if all(res.values()) else 1)


if __name__ == "__main__":
    main()
