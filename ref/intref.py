"""Целочисленный эталон инференса: ровно та арифметика, что будет на Z80.

Всё целое. Вещественные числа живут только в export(): там масштабы обученной модели
превращаются в целые множители и таблицы. Дальше forward_step() считает один символ для
пачки последовательностей сразу (numpy int64), и каждое промежуточное значение совпадает
с тем, что Z80 положит в память.

Соглашения (Z80 повторяет их буквально):
  * requant(acc, M, S) = (acc*M + 2^(S-1)) >> S, сдвиг арифметический (floor), потом clamp;
  * деление выхода внимания: o = sign(N) * ((8|N| + S/2) // S);
  * остаток кольцевого кэша: позиция t хранится в слоте t % ctx.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

EXP_UNIT = 16          # таблица exp: шаг 1/16 ната
SX = 32                # шаг int8 после RMSNorm
RS_BITS = 18           # точность таблицы обратного корня: K в [16416, 32768]


def exp_table(scale):
    return np.array([int(math.floor(scale * math.exp(-u / EXP_UNIT) + 0.5)) for u in range(256)], dtype=np.int64)


EXP63 = exp_table(63)      # вес позиции во внимании
EXP255 = exp_table(255)    # вероятности символов при выборке
# обратный корень для мантиссы m из [64, 256): K = round(2^RS_BITS / sqrt(m))
RSQRT = np.array([0] * 64 + [int(math.floor(2 ** RS_BITS / math.sqrt(m) + 0.5)) for m in range(64, 256)],
                 dtype=np.int64)


def requant(acc, M, S, lo, hi):
    return np.clip((acc * M + (1 << (S - 1))) >> S, lo, hi)


def mult_shift(f, mbits=8):
    """Вещественные множители f (>0) -> целые M < 2^mbits и общий сдвиг S."""
    f = np.asarray(f, dtype=np.float64)
    S = int(math.floor(math.log2((2 ** mbits - 1) / f.max())))
    M = np.floor(f * 2 ** S + 0.5).astype(np.int64)
    assert M.max() < 2 ** mbits and M.min() > 0, (M.min(), M.max())
    return M, S


@dataclass
class Layer:
    tq: np.ndarray            # (H*dh + 2*dh, d) триты q|k|v
    mqkv: np.ndarray          # множители по строкам
    sqkv: int
    to: np.ndarray            # (d, H*dh)
    mo: np.ndarray
    so: int
    t1: np.ndarray            # (ffn, d)
    m1: np.ndarray
    s1: int
    t2: np.ndarray            # (d, ffn)
    m2: np.ndarray
    s2: int
    bias: np.ndarray          # (H, ctx) добавка к счёту по расстоянию, целые
    act: np.ndarray           # (128,) таблица активации для pre8 >= 0
    # смесь экспертов: эксперт 0 — t1/t2 выше, остальные — (t1, m1, s1, t2, m2, s2); выбор —
    # argmax(acc_e * mr_e), acc = x8 . tr_e, при равенстве — меньший номер
    experts: list = None
    tr: np.ndarray = None
    mr: np.ndarray = None
    pin_bias: np.ndarray = None   # (H,) добавка к счёту закреплённых токенов начала


@dataclass
class IntModel:
    vocab: int
    d: int
    heads: int
    dh: int
    ffn: int
    ctx: int
    score_shift: int
    emb: np.ndarray           # (V, d): int7 (общий масштаб) или коды int4 -7..7
    emb_shift: int            # int7: h = emb << emb_shift
    layers: list = field(default_factory=list)
    logit_scale: float = 1.0  # логит в натах = logit_int * logit_scale (для оценки и выборки)
    final_shift: int = 8      # последняя норма: x8 = h*sx_final*8/|h|, sx_final = 2^(final_shift-3)
    final_lim: int = 127
    emb_mul: np.ndarray = None   # int4: h0 = код * emb_mul[токен] (16 бит)
    logit_row: np.ndarray = None  # int4: логит в натах = acc * logit_row[токен]
    data: str = ""            # tok512 — словарь BPE
    # входные таблицы (int4, как вложения): h0 += prev[предыдущий] * prev_mul + big[строка пары] * big_mul,
    # строка пары = (h1[предыдущий] ^ h2[текущий]) & (bigram - 1)
    prev: np.ndarray = None
    prev_mul: np.ndarray = None
    big: np.ndarray = None
    big_mul: np.ndarray = None
    h1: np.ndarray = None
    h2: np.ndarray = None
    pin: int = 0              # первые pin позиций сказки остаются в кэше внимания навсегда


def export(ckpt) -> IntModel:
    import torch
    sys_path_model()
    from model import Cfg, TinyGPT, cfg_from
    cfg = cfg_from(ckpt["cfg"])
    m = TinyGPT(cfg)
    m.load_state_dict(ckpt["model"])
    m.eval()
    assert cfg.quant
    with torch.no_grad():
        e = m.emb.detach()
        hmax = residual_max(m, cfg, ckpt.get("data", ""))
        if cfg.emb_q == "int4":
            # коды -7..7 и масштаб строки; единица остатка — чтобы и max|h| (с запасом x2) влез в int16,
            # и множитель строки 7*M не переполнил 16 бит
            codes, step = m.emb_codes()
            step = step.squeeze(1).numpy().astype(np.float64)
            tables = []                       # входные таблицы: (имя, коды, шаг строки)
            for name, w, ls, rd in (("prev", "prev", "prev_log_step", "prev_step_ready"),
                                    ("big", "big", "big_log_step", "big_step_ready")):
                if getattr(cfg, "prev_emb" if name == "prev" else "bigram", 0):
                    wq = m.table_q(getattr(m, w), getattr(m, ls), getattr(m, rd))
                    st = getattr(m, ls).exp().detach()
                    tables.append((name, torch.round(wq / st).clamp(-7, 7).long().numpy(),
                                   st.squeeze(1).numpy().astype(np.float64)))
            smax = max([step.max()] + [t[2].max() for t in tables])
            unit_h = max(hmax / 16384, smax / 4000)
            emb = codes.long().numpy()
            im = IntModel(cfg.vocab, cfg.d, cfg.heads, cfg.dh, cfg.ffn, cfg.ctx, cfg.score_shift, emb, 0)
            im.emb_mul = np.floor(step / unit_h + 0.5).astype(np.int64)
            for name, cd, st in tables:
                setattr(im, name, cd)
                setattr(im, name + "_mul", np.floor(st / unit_h + 0.5).astype(np.int64))
            if cfg.bigram:
                im.h1, im.h2 = m.h1.numpy().astype(np.int64), m.h2.numpy().astype(np.int64)
        else:
            s_e = float(e.abs().max() / cfg.emb_max)
            emb = torch.round(e / s_e).clamp(-cfg.emb_max, cfg.emb_max).long().numpy()
            # единица остаточного потока: s_e / 2^emb_shift; сдвиг — чтобы максимум |h| на
            # калибровочных окнах занимал не больше половины int16 (сложение в h — с насыщением)
            emb_shift = int(math.floor(math.log2(16384 * s_e / hmax)))
            emb_shift = max(0, min(emb_shift, 7))
            unit_h = s_e / 2 ** emb_shift
            im = IntModel(cfg.vocab, cfg.d, cfg.heads, cfg.dh, cfg.ffn, cfg.ctx, cfg.score_shift, emb, emb_shift)
        im.data = ckpt.get("data", "")
        im.pin = getattr(cfg, "pin", 0)
        for b in m.blocks:
            tq, aq = b.wq.ternary()
            tk, ak = b.wk.ternary()
            tv, av = b.wv.ternary()
            g = b.q_gain.detach().repeat_interleave(cfg.dh)
            f_q = aq * g * 16.0 / SX
            f_k = ak / (SX * b.fk.step)
            f_v = av / (SX * b.fv.step)
            mqkv, sqkv = mult_shift(torch.cat([f_q, f_k, f_v]).numpy())
            to, ao = b.wo.ternary()
            mo, so = mult_shift((ao * (b.fv.step / 8.0) / unit_h).numpy())
            t1, a1 = b.w1.ternary()
            m1, s1 = mult_shift((a1 / (SX * b.fpre.step)).numpy())
            t2, a2 = b.w2.ternary()
            m2, s2 = mult_shift((a2 * b.fpost.step / unit_h).numpy())
            pre = np.arange(128) * b.fpre.step
            post = {"relu2": np.maximum(pre, 0) ** 2, "relu": np.maximum(pre, 0)}[cfg.act]
            act = np.clip(np.floor(post / b.fpost.step + 0.5), 0, 127).astype(np.int64)
            bias = torch.round(b.rel_bias.detach()).long().numpy()
            ly = Layer(torch.cat([tq, tk, tv]).long().numpy(), mqkv, sqkv,
                       to.long().numpy(), mo, so, t1.long().numpy(), m1, s1, t2.long().numpy(), m2, s2, bias, act)
            if getattr(cfg, "experts", 1) > 1:
                ly.experts = []
                for w1, w2 in zip(b.ew1, b.ew2):
                    e1, a1 = w1.ternary()
                    e2, a2 = w2.ternary()
                    ly.experts.append((e1.long().numpy(), *mult_shift((a1 / (SX * b.fpre.step)).numpy()),
                                       e2.long().numpy(), *mult_shift((a2 * b.fpost.step / unit_h).numpy())))
                trc, ar = b.router.ternary()
                ly.tr = trc.long().numpy()
                ly.mr, _ = mult_shift(ar.numpy())
            if getattr(cfg, "pin", 0):
                ly.pin_bias = torch.round(b.pin_bias.detach()).long().numpy()
            im.layers.append(ly)
        sxf = getattr(cfg, "sx_final", SX)
        im.final_shift = int(round(math.log2(sxf))) + 3
        im.final_lim = 63 if sxf == 16 else 127
        if cfg.emb_q == "int4":
            im.logit_row = step / sxf * float(m.logit_gain)
        else:
            im.logit_scale = s_e / sxf * float(m.logit_gain)
    return im


def residual_max(m, cfg, data_name="", n=256, seed=0):
    """Максимум |h| в остаточном потоке QAT-модели на n случайных окнах train."""
    import torch
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent / "data"
    calib = root / f"{data_name}_calib.npy"           # те же окна, сохранённые заранее: экспорт без корпуса
    if data_name and calib.exists() and (n, seed) == (256, 0):
        win = np.load(calib)
        assert win.shape == (n, cfg.ctx)
    else:
        data = (np.memmap(root / f"{data_name}_train.u16", dtype=np.uint16, mode="r") if data_name
                else np.memmap(root / "train.u8", dtype=np.uint8, mode="r"))
        rng = np.random.default_rng(seed)
        ix = rng.integers(0, len(data) - cfg.ctx, n)
        win = np.asarray(data[ix[:, None] + np.arange(cfg.ctx)[None]])
    idx = torch.from_numpy(win.astype(np.int64))
    h = m.embed(idx)
    mx = float(h.abs().max())
    for b in m.blocks:
        h = h + b.attention(h)
        mx = max(mx, float(h.abs().max()))
        h = h + b.mlp(h)
        mx = max(mx, float(h.abs().max()))
    return mx


def sys_path_model():
    import sys
    from pathlib import Path
    p = str(Path(__file__).resolve().parent.parent / "train")
    if p not in sys.path:
        sys.path.insert(0, p)


def bitlen(x):
    """Длина модуля в битах, поэлементно (0 -> 0)."""
    x = np.abs(x)
    out = np.zeros_like(x)
    while np.any(x > 0):
        nz = x > 0
        out[nz] += 1
        x = x >> 1
    return out


NORM_HBITS = 7        # h' после сдвига — знаковый байт (замер: +0.0005 bpc против 12 бит)
NORM_KBITS = 8        # обратный корень K в [64, 128] — умножение 8x8


def rmsnorm8(h, out_shift=8, lim=127):
    """h (B, d) int16 -> x8 (B, d) в [-127, 127]. Как на Z80:
    s — сдвиг до 12 бит модуля, ssq = sum(h'^2) < 2^30, мантисса m = ssq >> e в [64, 256),
    e чётное; x8 = round(h' * K / 2^(RS_BITS + e/2 - 8))  (SX * sqrt(64) = 2^8)."""
    B, d = h.shape
    assert d == 64
    mx = np.abs(h).max(axis=1)
    s = np.maximum(bitlen(mx) - NORM_HBITS, 0)
    hp = h >> s[:, None]
    ssq = (hp * hp).sum(axis=1)
    e = np.zeros(B, dtype=np.int64)
    # e — наименьшее чётное, при котором ssq >> e < 256
    bl = bitlen(ssq)
    e = np.maximum(bl - 8 + 1, 0) // 2 * 2
    m = ssq >> e
    # маленькие ssq (< 64) — только при почти нулевом h; берём m >= 64 сдвигом влево
    m = np.maximum(m, 64)     # почти нулевой h; на обученной модели не встречается
    K = RSQRT[m]
    kd = 16 - NORM_KBITS
    if kd:
        K = (K + (1 << (kd - 1))) >> kd
    sh = RS_BITS - kd + e // 2 - out_shift
    return np.clip((hp * K[:, None] + (1 << (sh - 1))[:, None]) >> sh[:, None], -lim, lim)


ACC_MAX = {}


def note_acc(name, acc):
    ACC_MAX[name] = max(ACC_MAX.get(name, 0), int(np.abs(acc).max()))


class Runner:
    """Пачка последовательностей с кольцевым KV-кэшем."""

    def __init__(self, im: IntModel, B: int):
        self.im = im
        self.B = B
        L = len(im.layers)
        self.k = np.zeros((L, B, im.ctx, im.dh), dtype=np.int64)
        self.v = np.zeros((L, B, im.ctx, im.dh), dtype=np.int64)
        P = max(im.pin, 1)
        self.pk = np.zeros((L, B, P, im.dh), dtype=np.int64)       # закреплённые: позиции 0..pin-1
        self.pv = np.zeros((L, B, P, im.dh), dtype=np.int64)
        self.pos = 0
        self.prev_tok = np.zeros(B, dtype=np.int64)    # предыдущий токен (в начале — конец сказки)
        self.trace = None      # dict для сверки с Z80: имя -> массив

    def tr(self, name, val):
        if self.trace is not None:
            self.trace[name] = val.copy()

    def step(self, tok):
        im = self.im
        H, dh, ctx = im.heads, im.dh, im.ctx
        if im.emb_mul is not None:
            h = im.emb[tok] * im.emb_mul[tok][:, None]
        else:
            h = im.emb[tok] << im.emb_shift
        pt = self.prev_tok
        if im.prev is not None:
            h = h + im.prev[pt] * im.prev_mul[pt][:, None]
        if im.big is not None:
            row = (im.h1[pt] ^ im.h2[tok]) & (len(im.big) - 1)
            h = h + im.big[row] * im.big_mul[row][:, None]
        self.prev_tok = np.asarray(tok, dtype=np.int64).copy()
        assert np.abs(h).max() < 32768
        self.tr("h0", h)
        n = min(self.pos + 1, ctx)
        slots = [(self.pos - i) % ctx for i in range(n)]       # слот позиции pos-i
        dist = np.arange(n)
        for li, ly in enumerate(im.layers):
            x = rmsnorm8(h)
            self.tr(f"l{li}.x_att", x)
            acc = x @ ly.tq.T
            note_acc(f"l{li}.qkv", acc)
            assert np.abs(acc).max() < 32768
            qkv = requant(acc, ly.mqkv, ly.sqkv, -63, 63)
            self.tr(f"l{li}.qkv", qkv)
            q = qkv[:, :H * dh].reshape(-1, H, dh)
            slot = self.pos % ctx
            self.k[li, :, slot] = qkv[:, H * dh:H * dh + dh]
            self.v[li, :, slot] = qkv[:, H * dh + dh:]
            if self.pos < im.pin:
                self.pk[li, :, self.pos] = self.k[li, :, slot]
                self.pv[li, :, self.pos] = self.v[li, :, slot]
            kk = self.k[li][:, slots]                             # B n dh
            vv = self.v[li][:, slots]
            raw = np.einsum("bhd,bnd->bhn", q, kk)
            s = (raw >> im.score_shift) + ly.bias[None, :, dist]
            if im.pin:                                            # закреплённые 0..min(pos, pin-1) — после окна
                npin = min(self.pos + 1, im.pin)
                kp, vp = self.pk[li][:, :npin], self.pv[li][:, :npin]
                sp = (np.einsum("bhd,bnd->bhn", q, kp) >> im.score_shift) + ly.pin_bias[None, :, None]
                s = np.concatenate([s, sp], axis=2)
                vv = np.concatenate([vv, vp], axis=1)
            u = np.minimum(s.max(axis=2, keepdims=True) - s, 255)
            e = EXP63[u]
            S = e.sum(axis=2)                                    # B H
            N = np.einsum("bhn,bnd->bhd", e, vv)
            o = np.sign(N) * ((8 * np.abs(N) + (S // 2)[:, :, None]) // S[:, :, None])
            o = o.reshape(-1, H * dh)
            self.tr(f"l{li}.o", o)
            acc = o @ ly.to.T
            note_acc(f"l{li}.o", acc)
            assert np.abs(acc).max() < 32768
            h = np.clip(h + requant(acc, ly.mo, ly.so, -32768, 32767), -32768, 32767)
            self.tr(f"l{li}.h_att", h)
            x = rmsnorm8(h)
            mats = [(ly.t1, ly.m1, ly.s1, ly.t2, ly.m2, ly.s2)] + (ly.experts or [])
            if ly.experts:
                racc = x @ ly.tr.T
                assert np.abs(racc).max() < 32768
                sel = np.argmax(racc * ly.mr[None], axis=1)
                self.tr(f"l{li}.expert", sel)
            else:
                sel = np.zeros(len(x), dtype=np.int64)
            out = np.zeros_like(h)
            for ei, (t1, m1, s1, t2, m2, s2) in enumerate(mats):
                rows = sel == ei
                if not rows.any():
                    continue
                acc = x[rows] @ t1.T
                note_acc(f"l{li}.w1", acc)
                assert np.abs(acc).max() < 32768
                pre = requant(acc, m1, s1, -127, 127)
                post = np.where(pre > 0, ly.act[np.maximum(pre, 0)], 0)
                acc = post @ t2.T
                note_acc(f"l{li}.w2", acc)
                assert np.abs(acc).max() < 32768
                out[rows] = requant(acc, m2, s2, -32768, 32767)
            h = np.clip(h + out, -32768, 32767)
            self.tr(f"l{li}.h_mlp", h)
        x = rmsnorm8(h, im.final_shift, im.final_lim)
        logits = x @ im.emb.T
        note_acc("logits", logits)
        self.tr("logits", logits)
        self.pos += 1
        return logits


def eval_bpc(im: IntModel, data: np.ndarray, n_windows=1024, T=64, B=256, piece_len=None):
    """Биты на символ на тех же окнах valid, что и train.evaluate (первые n_windows окон длины T)."""
    tot, cnt = 0.0, 0
    for i in range(0, n_windows, B):
        nb = min(B, n_windows - i)
        starts = (np.arange(i, i + nb) * T)
        chunk = data[starts[:, None] + np.arange(T + 1)[None]].astype(np.int64)
        r = Runner(im, nb)
        r.prev_tok = np.where(starts > 0, data[np.maximum(starts - 1, 0)], 0).astype(np.int64)   # как train.evaluate
        for t in range(T):
            lg = r.step(chunk[:, t])
            lg = lg * (im.logit_row[None] if im.logit_row is not None else im.logit_scale)
            lg = lg - lg.max(axis=1, keepdims=True)
            lp = lg - np.log(np.exp(lg).sum(axis=1, keepdims=True))
            tot += -lp[np.arange(nb), chunk[:, t + 1]].sum()
            cnt += int(piece_len[chunk[:, t + 1]].sum()) if piece_len is not None else nb
    return tot / cnt / math.log(2)
