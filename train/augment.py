"""Корпус с подменой героев и предметов: модель должна брать их из текста, а не из памяти.

    python train/augment.py lists              -> data/aug_lists.json (существительные и имена сказок, пулы замен)
    python train/augment.py corpus [доля=0.5]  -> data/tok512aug_{train,valid}.u16
    python train/augment.py speccy             -> data/speccy512aug_{train,valid}.u16 (только имена)

В выбранной сказке каждое имя из частых имён корпуса меняется на случайное имя того же пола, а одно-два
самых частых в ней существительных — на случайные предметные существительные WordNet; замена сквозная
по всей сказке (вместе с множественным числом, заглавной буквой и артиклем a/an).
"""
from __future__ import annotations

import json
import random
import re
import sys
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bpe  # noqa: E402
from prep_data import VOCAB  # noqa: E402

DATA = bpe.DATA
LISTS = DATA / "aug_lists.json"
CODE = np.full(256, 255, dtype=np.uint8)
for i, ch in enumerate(VOCAB):
    CODE[ord(ch)] = i


def plural(w: str) -> str:
    if re.search(r"[^aeiou]y$", w):
        return w[:-1] + "ies"
    if re.search(r"(s|x|z|ch|sh)$", w):
        return w + "es"
    return w + "s"


def make_lists():
    import nltk
    from nltk.corpus import names, wordnet as wn
    from wordfreq import zipf_frequency
    src = np.memmap(DATA / "train.u8", dtype=np.uint8, mode="r")
    stories = bpe.ids_to_text(np.asarray(src[:30_000_000])).split("\0")[1:-1]
    random.Random(1).shuffle(stories)
    nn, other, cap = Counter(), Counter(), Counter()
    t0 = time.time()
    for s in stories[:20000]:
        for sent in re.split(r"(?<=[.!?])\s+", s):
            toks = re.findall(r"[A-Za-z]+", sent)
            for i, (w, tag) in enumerate(nltk.pos_tag(toks)):
                if tag == "NN" and w.islower():
                    nn[w] += 1
                elif w.islower():
                    other[w] += 1
                if i > 0 and w[0].isupper() and tag == "NNP":
                    cap[w] += 1
    print(f"разметка {time.time() - t0:.0f} с", flush=True)
    male = {w for w in names.words("male.txt") if w.isalpha() and 3 <= len(w) <= 9}
    female = {w for w in names.words("female.txt") if w.isalpha() and 3 <= len(w) <= 9}
    roots = [wn.synset(x) for x in ("animal.n.01", "artifact.n.01", "food.n.01", "food.n.02", "plant.n.02",
                                     "natural_object.n.01", "geological_formation.n.01", "body_of_water.n.01")]
    person = wn.synset("person.n.01")

    def kind(w, first_only):
        syns = wn.synsets(w, "n")[:1 if first_only else 3]
        for syn in syns:
            hyp = set(syn.closure(lambda t: t.hypernyms())) | {syn}
            if any(r in hyp for r in roots):
                return "thing"
            if person in hyp:
                return "person"
        return None
    bad = set("arse ass bitch slut whore dick cock penis vagina crap shit bastard sex drug cocaine heroin "
              "condom bra panties corpse coffin bomb rifle pistol cigarette cigar beer whiskey vodka wine".split())
    ts_nouns = [w for w, n in nn.most_common() if n >= 15 and n >= 4 * other[w] and len(w) >= 3
                and kind(w, True) == "thing"]
    skip_names = {"Mommy", "Daddy", "Mom", "Dad", "Cat", "Dog", "Bunny", "Bear", "Grandma", "Grandpa"}
    ts_names = [w for w, n in cap.most_common() if n >= 10 and (w in male or w in female) and w not in skip_names]
    pool = set()
    for syn in wn.all_synsets("n"):
        for lem in syn.lemmas():
            w = lem.name()
            if not (w.isalpha() and w.islower() and 3 <= len(w) <= 11) or w.endswith("ing") or w in bad:
                continue
            k = kind(w, True)
            if k == "thing" and zipf_frequency(w, "en") >= 2.8 or k == "person" and zipf_frequency(w, "en") >= 3.2:
                pool.add(w)
    pool = sorted(pool | set(ts_nouns))
    lists = dict(ts_nouns=ts_nouns, ts_names=ts_names, nouns=pool,
                 male=sorted(male - set(ts_names) | {"Tim", "Tom", "Ben", "Max", "Sam", "Jack", "Leo", "Bob"}),
                 female=sorted(female - set(ts_names) | {"Lily", "Sue", "Mia", "Lucy", "Anna", "Amy", "Kim", "Sara"}),
                 gender={w: ("f" if w in female and w not in male else "m") for w in ts_names})
    LISTS.write_text(json.dumps(lists), encoding="utf-8")
    print(f"существительных сказок {len(ts_nouns)}: {ts_nouns[:40]}")
    print(f"имён сказок {len(ts_names)}: {ts_names[:40]}")
    print(f"пул замен: {len(pool)} существительных, {len(lists['male'])} м. и {len(lists['female'])} ж. имён")
    print("примеры пула:", random.Random(2).sample(pool, 40))


_L = None
_NOUN_SET = None
_NAME_SET = None


def _init():
    global _L, _NOUN_SET, _NAME_SET
    _L = json.loads(LISTS.read_text(encoding="utf-8"))
    _NOUN_SET = set(_L["ts_nouns"])
    _NAME_SET = set(_L["ts_names"])


def _case(src: str, dst: str) -> str:
    return dst[0].upper() + dst[1:] if src[0].isupper() else dst


def augment_story(s: str, rng: random.Random, nouns=True) -> str:
    words = re.findall(r"[A-Za-z]+", s)
    sub = {}
    for w in sorted(set(words) & _NAME_SET):            # порядок не от соли хеша строк — иначе прогоны разные
        if not nouns:
            # сказки про Спекки: обычные имена из самих сказок того же пола — редкие имена учат выдумывать
            # несуществующие («Siobp», «Norvitation»)
            pool = [n for n in _L["ts_names"] if _L["gender"].get(n) == _L["gender"].get(w)]
        else:
            pool = _L["female"] if _L["gender"].get(w) == "f" else _L["male"]
        sub[w] = rng.choice(pool)
    if nouns:
        cnt = Counter()
        for w in words:
            lw = w.lower()
            if lw in _NOUN_SET:
                cnt[lw] += 1
            elif lw.endswith("s") and lw[:-1] in _NOUN_SET:
                cnt[lw[:-1]] += 1
        top = [w for w, n in cnt.most_common(2) if n >= 2]
        for w in top[:rng.choice((1, 2))]:
            new = rng.choice(_L["nouns"])
            for a, b in ((w, new), (plural(w), plural(new)), (w + "s", plural(new))):
                sub.setdefault(a, b)
                sub.setdefault(a.capitalize(), b.capitalize())
    if not sub:
        return s
    pat = re.compile(r"\b(" + "|".join(sorted(map(re.escape, sub), key=len, reverse=True)) + r")\b")
    out = pat.sub(lambda m: sub[m.group(1)], s)
    # артикль: a apple -> an apple, an dog -> a dog (только перед подменёнными словами)
    news = {v.lower() for v in sub.values()}

    def art(m):
        a, w = m.group(1), m.group(2)
        if w.lower() not in news:
            return m.group(0)
        an = w[0].lower() in "aeiou"
        want = ("An" if a[0] == "A" else "an") if an else ("A" if a[0] == "A" else "a")
        return want + " " + w
    return re.sub(r"\b([Aa]n?) ([A-Za-z]+)", art, out)


def _work(args):
    lo, hi, split, frac, seed, nouns = args
    src = np.memmap(DATA / f"{split}.u8", dtype=np.uint8, mode="r")
    text = bpe.ids_to_text(np.asarray(src[lo:hi]))
    rng = random.Random(seed)
    parts = text.split("\0")
    parts = [augment_story(p, rng, nouns) if p and rng.random() < frac else p for p in parts]
    text = "\0".join(parts)
    pieces = bpe.PRE.findall(text)
    ids = np.fromiter((t for p in pieces for t in _ENC(p)), dtype=np.uint16)
    return ids, len(text)


_ENC = None


def _init_all():
    global _ENC
    _init()
    meta = json.loads((DATA / "tok512.json").read_text(encoding="utf-8"))
    _ENC = bpe.encoder([tuple(m) for m in meta["merges"]], meta["pieces"])


def spans(src, ch=20_000_000):
    out, lo = [], 0
    while lo < len(src):
        hi = min(lo + ch, len(src))
        if hi < len(src):
            nz = np.flatnonzero(np.asarray(src[hi:hi + 1_000_000]) == 0)
            hi = hi + int(nz[0]) + 1 if len(nz) else hi
        out.append((lo, hi))
        lo = hi
    return out


def build(split_in, name_out, frac, nouns=True, procs=8, seed0=1000):
    src = np.memmap(DATA / f"{split_in}.u8", dtype=np.uint8, mode="r")
    jobs = [(lo, hi, split_in, frac, seed0 + i, nouns) for i, (lo, hi) in enumerate(spans(src))]
    tmp = DATA / f"{name_out}.tmp"
    t0, toks = time.time(), 0
    with open(tmp, "wb") as f, Pool(procs, initializer=_init_all) as pool:
        for ids, n in pool.imap(_work, jobs):
            ids.tofile(f)
            toks += len(ids)
            print(f"{name_out}: {toks / 1e6:.0f}M токенов ({time.time() - t0:.0f} с)", flush=True)
    tmp.replace(DATA / name_out)


def main():
    cmd = sys.argv[1]
    if cmd == "lists":
        make_lists()
    elif cmd == "demo":
        _init()
        rng = random.Random(int(sys.argv[2]) if len(sys.argv) > 2 else 0)
        src = np.memmap(DATA / "valid.u8", dtype=np.uint8, mode="r")
        for s in bpe.ids_to_text(np.asarray(src[:20000])).split("\0")[1:4]:
            print(augment_story(s, rng), "\n----")
    elif cmd == "corpus":
        frac = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
        build("valid", "tok512aug_valid.u16", frac)
        build("train", "tok512aug_train.u16", frac)
    elif cmd == "speccy":
        # истории про Спекки: только имена (слова Спектрума не трогаем)
        for split in ("train", "valid"):
            a = np.fromfile(DATA / f"speccy512_{split}.u16", dtype=np.uint16)
            meta = json.loads((DATA / "tok512.json").read_text(encoding="utf-8"))
            text = "".join(meta["pieces"][t] for t in a.tolist())
            (DATA / f"speccy_{split}.u8").write_bytes(CODE[np.frombuffer(text.encode("ascii"), np.uint8)].tobytes())
            # 16 копий с разными именами: сказок всего полторы тысячи, при дообучении каждая идёт много раз
            copies = 16 if split == "train" else 1
            parts = []
            for c in range(copies):
                build(f"speccy_{split}", f"speccy512aug_{split}.u16", 0.9, nouns=False, procs=1, seed0=7000 + 100 * c)
                parts.append(np.fromfile(DATA / f"speccy512aug_{split}.u16", dtype=np.uint16))
            np.concatenate(parts).tofile(DATA / f"speccy512aug_{split}.u16")


if __name__ == "__main__":
    main()
