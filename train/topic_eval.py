"""Держится ли модель темы затравки: выборка продолжений и простые счётчики.

    python train/topic_eval.py <ckpt> [<ckpt> ...] [--k 16] [--temp 0.6] [--toks 50] [--show 2]

Для каждой затравки — k продолжений по toks токенов (как на Спектруме: перед затравкой конец сказки,
последний токен затравки склеивается с первым выбранным). Счётчики по продолжению:
  keep    — встретилось хотя бы одно ключевое слово затравки (по началу слова, без регистра);
  restart — «Once upon a time» или конец сказки в первых 20 токенах: модель бросила затравку;
  intrude — в первых двух предложениях всплыл чужой герой из частых имён корпуса или Speccy/Hippo/rainbow,
            которых нет в затравке.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bpe  # noqa: E402
from model import TinyGPT, cfg_from  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

# (затравка, ключевые слова). Три группы: знакомые слова сказок, незнакомые, Спектрум.
PROMPTS = [
    ("The little dog found a bone in the garden.", "dog bone"),
    ("A big red fire truck", "truck fire"),
    ("The princess lost her crown", "princess crown"),
    ("The old wizard had a magic hat", "wizard hat magic"),
    ("A robot named Bolt", "robot bolt"),
    ("The pirate ship sailed", "pirate ship"),
    ("The dragon was hungry", "dragon"),
    ("The bear wanted honey", "bear honey"),
    ("The farmer had a cow", "farmer cow"),
    ("The train went very fast", "train"),
    ("Grandma baked a cake", "grandma cake"),
    ("The snowman was melting", "snowman snow"),
    ("The monkey stole a banana", "monkey banana"),
    ("A fish lived in the pond", "fish pond"),
    ("The doctor was very busy", "doctor"),
    ("The astronaut flew to the moon", "astronaut moon"),
    ("A penguin lived in the desert", "penguin desert"),
    ("Grandpa fixed his old tractor", "tractor grandpa"),
    ("The volcano woke up", "volcano"),
    ("The samurai had a sword", "samurai sword"),
    ("Kevin the koala", "kevin koala"),
    ("The yellow submarine", "submarine"),
    ("The ninja crept into the castle", "ninja castle"),
    ("The vampire was afraid of garlic", "vampire garlic"),
    ("The haunted house on the hill", "haunted house"),
    ("The giraffe played the violin", "giraffe violin"),
    ("My grandmother had a samovar", "samovar grandmother"),
    ("The octopus opened a bakery", "octopus bakery"),
    ("The king of the mushrooms", "king mushroom"),
    ("The tiny dinosaur", "dinosaur"),
    ("A snail raced a turtle", "snail turtle"),
    ("The lighthouse keeper", "lighthouse keeper"),
    ("The witch's broom broke", "witch broom"),
    ("The cowboy rode his horse", "cowboy horse"),
    ("The wolf and the seven goats", "wolf goat"),
    ("Captain Nemo", "nemo captain"),
    ("The clock stopped at midnight", "clock midnight"),
    ("The magic carpet", "carpet"),
    ("An alien landed in the park", "alien"),
    ("Once upon a time, there was a robot", "robot"),
    ("Once upon a time, a penguin", "penguin"),
    ("The scientist built a time machine", "scientist machine"),
    ("Dizzy the egg", "dizzy egg"),
    ("The ZX Spectrum", "spectrum speccy computer"),
    ("R Tape loading error", "tape error load"),
    ("Manic Miner", "miner willy mine"),
    ("The joystick was broken", "joystick"),
    ("The old cassette", "cassette tape"),
]
NAMES = set("Tim Timmy Lily Tom Ben Sue Max Sam Mia Lucy Anna Leo Jack Kim Amy Bob Rosa Emma Sara Zoe Willy "
            "Speccy Hippo".split())


def load(path, dev):
    ck = torch.load(path, map_location=dev)
    cfg = cfg_from(ck["cfg"])
    m = TinyGPT(cfg).to(dev)
    m.load_state_dict(ck["model"])
    m.eval()
    return m, cfg, ck.get("data") or "tok512"


STOP = set("the a an and was were had has his her their its of in on at to with for from into named old big little "
           "small once upon time there one day very".split())


HEAL_WORDS = True


def boost_tokens(prompt, enc):
    """Первые токены значимых слов затравки (со пробелом впереди и с заглавной) — им прибавка к логиту."""
    out = set()
    for w in re.findall(r"[A-Za-z]+", prompt):
        if len(w) >= 3 and w.lower() not in STOP:
            for v in (" " + w.lower(), " " + w.capitalize()):
                out.add(enc(v)[0])
    return sorted(out)


@torch.no_grad()
def continue_prompt(m, cfg, prompt, enc, pieces, k, temp, n_tok, gen, dev, boost=0.0):
    bt = boost_tokens(prompt, enc) if boost else []
    ids = [0] + [t for p in bpe.PRE.findall(prompt) for t in enc(p)]
    need = ids.pop()
    allowed = torch.tensor([pc.startswith(pieces[need]) for pc in pieces], device=dev)
    if HEAL_WORDS is False and re.search("[A-Za-z]", pieces[need]):
        ids.append(need)            # затравка кончается словом: слово законченное, склейки нет,
        need = None                 # и первый токен не продолжает его буквами
        allowed = torch.tensor([not (pc[:1].isalpha()) for pc in pieces], device=dev)
    x = torch.tensor(ids, device=dev)[None].repeat(k, 1)
    alive = torch.ones(k, dtype=torch.bool, device=dev)
    out = [[] for _ in range(k)]
    for step in range(n_tok):
        p0 = x[:, -cfg.ctx - 1:-cfg.ctx] if x.shape[1] > cfg.ctx else None
        pin = am = None
        if getattr(cfg, "pin", 0):
            P, T = cfg.pin, min(x.shape[1], cfg.ctx)
            pin = x[:, :P]
            if pin.shape[1] < P:
                pin = torch.cat([pin, pin.new_zeros(k, P - pin.shape[1])], 1)
            sp = x.shape[1] - T + torch.arange(T, device=dev)              # место позиции окна в сказке
            am = (torch.arange(P, device=dev)[None, None] <= sp[None, :, None]).expand(k, T, P)
        logits = m(x[:, -cfg.ctx:], prev0=p0, pin=pin, amask=am)[:, -1].float() / temp
        if bt:
            logits[:, bt] += boost / temp
        if step == 0:
            logits = logits.masked_fill(~allowed, float("-inf"))
        t = torch.multinomial(torch.softmax(logits, -1), 1, generator=gen)
        for i in range(k):
            if alive[i]:
                out[i].append(int(t[i]))
        alive &= t[:, 0] != 0
        if not alive.any():
            break
        x = torch.cat([x, t], 1)
    head = prompt[:len(prompt) - len(pieces[need])] if need is not None else prompt
    texts = []
    for o in out:
        s = "".join(pieces[t] for t in o if t != 0)
        texts.append((head + s, o))
    return texts


def score(prompt, keys, text, toks, pieces=None):
    cont = text[len(prompt):]
    low = cont.lower()
    keep = any(re.search(r"\b" + re.escape(kw), low) for kw in keys.split())
    late = None
    if pieces is not None and len(toks) > 80:     # ключевое слово в токенах 70+ (затравка уже за окном 64)
        tail = "".join(pieces[t] for t in toks[70:] if t != 0).lower()
        late = any(re.search(r"\b" + re.escape(kw), tail) for kw in keys.split())
    first20 = toks[:20]
    restart = ("once upon a time" in low and "once upon" not in prompt.lower()) or (0 in first20)
    sents = re.split(r"(?<=[.!?])\s", cont)
    head2 = " ".join(sents[:2])
    pnames = set(re.findall(r"[A-Z][a-z]+", prompt))
    intr = any(w in NAMES and w not in pnames for w in re.findall(r"[A-Z][a-z]+", head2)) or \
        ("rainbow" in head2.lower() and "rainbow" not in prompt.lower())
    return keep, restart, intr, late


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--temp", type=float, default=0.6)
    ap.add_argument("--toks", type=int, default=50)
    ap.add_argument("--show", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--json", type=str, default="")
    ap.add_argument("--boost", type=float, default=0.0, help="прибавка к логиту первых токенов слов затравки")
    ap.add_argument("--no-heal-words", action="store_true", help="не склеивать, если затравка кончается буквой")
    a = ap.parse_args()
    global HEAL_WORDS
    HEAL_WORDS = not a.no_heal_words
    dev = "cuda"
    res = {}
    for path in a.ckpts:
        m, cfg, data = load(path, dev)
        meta = json.loads((ROOT / "data" / f"{data}.json").read_text(encoding="utf-8"))
        enc = bpe.encoder([tuple(x) for x in meta["merges"]], meta["pieces"])
        gen = torch.Generator(device=dev).manual_seed(a.seed)
        tot = [0, 0, 0]
        per = []
        lates = []
        samples = []
        for prompt, keys in PROMPTS:
            outs = continue_prompt(m, cfg, prompt, enc, meta["pieces"], a.k, a.temp, a.toks, gen, dev, a.boost)
            sc = [score(prompt, keys, t, o, meta["pieces"]) for t, o in outs]
            late = [c[3] for c in sc if c[3] is not None]
            lates.append((sum(late), len(late)))
            s = [sum(c[j] for c in sc) for j in range(3)]
            per.append((prompt, *[v / a.k for v in s]))
            tot = [x + y for x, y in zip(tot, s)]
            samples.append((prompt, [t for t, _ in outs]))
        n = a.k * len(PROMPTS)
        name = (Path(path).parent.name if Path(path).name == "last.pt" else Path(path).stem) + (f"+boost{a.boost:g}" if a.boost else "")
        res[name] = dict(keep=tot[0] / n, restart=tot[1] / n, intrude=tot[2] / n,
                         per={p: dict(keep=k, restart=r, intrude=i) for p, k, r, i in per}, samples=samples)
        ln, ld = sum(x for x, _ in lates), sum(y for _, y in lates)
        res[name]["late"] = ln / ld if ld else None
        print(f"{name:28s} keep {tot[0] / n:.3f}  restart {tot[1] / n:.3f}  intrude {tot[2] / n:.3f}"
              + (f"  late {ln / ld:.3f} ({ld})" if ld else ""), flush=True)
        for prompt, texts in samples[:: max(1, len(samples) // 8)] if a.show else []:
            for t in texts[:a.show]:
                print("   ", t.replace("\n", " / ")[:220])
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
