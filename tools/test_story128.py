"""HippoGPT 128K целиком на стенде: разбор затравки, история токен за токеном, снимок экрана.

    python tools/test_story128.py ["затравка"] [зерно] [цифра температуры] [--256]

--256 — сборка для Scorpion (build256, стенд zx256.Machine256, без склейки законченного слова).

Сверяет с эталоном (ref/intref.py + ref/sampler.py) разбор затравки (слияния BPE — на строках из корпуса)
и каждый выбранный токен, включая склейку последнего токена затравки.
Экран сохраняет в build/story.png (шапка — art/out/hippogpt.scr).
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ref"))
import intref  # noqa: E402
import sampler  # noqa: E402
from zx128 import BUILD, ROOT, Machine128  # noqa: E402

PAL = np.array([[0, 0, 0], [0, 0, 215], [215, 0, 0], [215, 0, 215], [0, 215, 0], [0, 215, 215], [215, 215, 0],
                [215, 215, 215], [0, 0, 0], [0, 0, 255], [255, 0, 0], [255, 0, 255], [0, 255, 0], [0, 255, 255],
                [255, 255, 0], [255, 255, 255]], dtype=np.uint8)


def screen_png(scr: bytes, path, scale=3):
    pix = np.zeros((192, 256), dtype=bool)
    for y in range(192):
        addr = ((y & 0xC0) << 5) | ((y & 7) << 8) | ((y & 0x38) << 2)
        pix[y] = np.unpackbits(np.frombuffer(scr[addr:addr + 32], dtype=np.uint8))
    att = np.frombuffer(scr[6144:6912], dtype=np.uint8).reshape(24, 32)
    ink = PAL[(att & 7) + ((att >> 6) & 1) * 8].repeat(8, 0).repeat(8, 1)
    paper = PAL[((att >> 3) & 7) + ((att >> 6) & 1) * 8].repeat(8, 0).repeat(8, 1)
    img = np.where(pix[:, :, None], ink, paper)
    border = np.zeros((192 + 48, 256 + 64, 3), dtype=np.uint8)
    border[24:216, 32:288] = img
    Image.fromarray(border).resize(((256 + 64) * scale, (192 + 48) * scale), Image.NEAREST).save(path)


def reference_story(im, pieces, merges, lm, tq, prompt, seed, heal_words, min_eos=0, max_tok=200):
    """Эталон (ref/intref.py): токены истории без конца сказки, как их пишет GENLOG."""
    r = intref.Runner(im, 1)
    ids = sampler.tokenize_bpe(prompt, merges)
    need = ids.pop() if ids else None              # склейка: последний токен — не в модель
    nocont = False
    if need is not None and not heal_words and prompt[-1].isalpha():
        ids.append(need)                           # законченное слово: склейки нет,
        need = None                                # и первый токен не продолжает его буквами
        nocont = True
    logits = r.step(np.array([0]))[0]
    for tok in ids:
        logits = r.step(np.array([tok]))[0]
    rng, ref = seed, []
    for _ in range(max_tok):
        if len(ref) < min_eos:
            logits = logits.copy()
            logits[0] = sampler.HEAL_MASK                 # конец сказки пока нельзя
        if nocont:
            logits = logits.copy()
            logits[[k for k, pc in enumerate(pieces) if pc[:1].isalpha()]] = sampler.HEAL_MASK
            nocont = False
        if need is not None:
            logits, need0, need = sampler.heal_mask(logits, pieces, need), need, None
            assert need0 is not None
        tok, rng = sampler.sample(logits, lm, tq, rng)
        if tok == 0:
            break
        ref.append(tok)
        logits = r.step(np.array([tok]))[0]
    return ref


def main():
    global BUILD
    s256 = "--256" in sys.argv
    argv = [a for a in sys.argv if a != "--256"]
    prompt = argv[1] if len(argv) > 1 else "Once upon a time"
    seed = int(argv[2], 0) if len(argv) > 2 else 0xACE1
    tkey = argv[3] if len(argv) > 3 else None
    if s256:
        import zx256
        BUILD = zx256.BUILD
    with open(BUILD / ("intmodel256.pkl" if s256 else "intmodel128.pkl"), "rb") as f:
        im = pickle.load(f)
    meta = json.loads((ROOT / "data" / f"{im.data}.json").read_text(encoding="utf-8"))
    pieces, merges = meta["pieces"], [tuple(x) for x in meta["merges"]]
    lm, unit_z = sampler.logit_mults(im.logit_row)
    scr = (ROOT / "art" / "out" / "hippogpt.scr").read_bytes()
    m = zx256.Machine256(screen=scr) if s256 else Machine128(screen=scr)
    S = m.syms
    heal_words = S.get("HEAL_WORDS", 1) if "HEAL_WORDS" in S else (0 if s256 else 1)

    # --- разбор затравки: особые строки + куски корпуса по 29 символов ---
    texts = ["Once upon a time", "The little Speccy!", "Lily and Tom, \"Hello!\" said Max.", "a  b", " ", "x",
             "R Tape loading error", "aaaa  bbb!!! ?", "The computer pressed loud keys."]
    corpus = (ROOT / "data" / "speccy" / "speccy_stories.txt").read_text(encoding="utf-8")
    corpus = corpus.replace("\n<|endoftext|>\n", " ").replace("\n", " ")
    texts += [corpus[i:i + 29] for i in range(0, 29 * 150, 29)]
    worst = 0
    for k, text in enumerate(texts):
        m.poke(S["PROMPT"], text.encode("ascii"))
        m.poke(S["PROMPT_LEN"], [len(text)])
        worst = max(worst, m.call("tokenize"))
        n = m.peek(S["T_N"], 1)[0]
        got = m.peek_words(S["TOKBUF"], n, signed=False)
        ref = sampler.tokenize_bpe(text, merges)
        if k < 9 or got != ref:
            print(f"разбор {'OK ' if got == ref else 'РАСХОДИТСЯ'} {text!r}: " + "|".join(pieces[t] for t in got))
        if got != ref:
            print("   эталон:", "|".join(pieces[t] for t in ref))
            sys.exit(1)
    print(f"разбор: {len(texts)} строк совпали с эталоном, дольше всего {worst / 3.5e6:.2f} с")

    # --- история ---
    m.call("init_sq")
    m.call("cls_chat")
    m.call("draw_temp")
    if tkey is not None:
        m.call("set_temp", a=ord(tkey))
    tq = m.peek_words(S["TQ"], 1, signed=False)[0]
    key = chr(m.peek(S["TEMP_KEY"], 1)[0])
    temp = (int(key) or 10) / 10
    assert tq == sampler.temp_q(unit_z, temp), (key, tq)
    print(f"температура {temp} (клавиша {key}), TQ = {tq}")
    m.poke(S["TXT_ATTR"], [0x46])
    m.call("puts", hl=S["MSG_HELLO"])
    m.call("newline_soft")
    m.poke(S["TXT_ATTR"], [0x47])
    for ch in "> " + prompt:
        m.call("putc", a=ord(ch))
    m.poke(S["PROMPT"], prompt.encode("ascii"))
    m.poke(S["PROMPT_LEN"], [len(prompt)])
    m.poke_words(S["RND"], [seed])
    m.poke_words(S["GEN_N"], [0])
    t0 = time.time()
    t = m.call("story")
    dt = time.time() - t0
    n = m.peek_words(S["GEN_N"], 1, signed=False)[0]
    m.bank(7)
    got = m.peek_words(S["GENLOG"], n, signed=False)

    ref = reference_story(im, pieces, merges, lm, tq, prompt, seed, heal_words, S.get("MIN_EOS", 0))
    same = got == ref
    print(f"история: {'OK' if same else 'РАСХОДИТСЯ'}, {n} токенов, {t / max(n, 1) / 1e6:.2f} млн тактов на токен "
          f"(= {t / max(n, 1) / 3.5e6:.2f} с), симулятор {dt:.1f} с")
    if not same:
        k = next((i for i, (a, b) in enumerate(zip(got, ref)) if a != b), min(len(got), len(ref)))
        print(f"   первое расхождение на токене {k}: z80 {got[k:k + 5]} эталон {ref[k:k + 5]}")
    healed = sampler.tokenize_bpe(prompt, merges)[-1:] if prompt else []
    if healed and not heal_words and prompt[-1].isalpha():
        healed = []
    head = prompt[:len(prompt) - len(pieces[healed[0]])] if healed else prompt
    print(head + "".join(pieces[t] for t in got))
    m.bank(5)
    screen_png(bytes(m.peek(0x4000, 6912)), BUILD / "story.png")
    print("экран:", BUILD / "story.png")
    sys.exit(0 if same else 1)


if __name__ == "__main__":
    main()
