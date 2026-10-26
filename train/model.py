"""Крошечный посимвольный трансформер с тернарными весами и целочисленными активациями.

Обучение с учётом квантования (QAT): прямой проход повторяет то, что потом посчитает Z80,
обратный идёт сквозь округления как сквозь тождество (STE).

  * веса линейных слоёв — триты {-1, 0, +1} с масштабом на строку (BitNet b1.58, порог —
    среднее модулей строки);
  * вход каждой матрицы — int8: после RMSNorm с постоянным масштабом SX, внутри блоков —
    с обучаемым шагом (LSQ);
  * позиция — только относительная: обучаемая добавка к счёту внимания по расстоянию
    (начало как у ALiBi), поэтому на машине работает кольцевой KV-кэш без конца;
  * K и V одни на все головы (MQA) — кэш вчетверо меньше;
  * вложения int8, вход и выход делят одну таблицу.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Cfg:
    vocab: int = 64
    d: int = 64
    layers: int = 2
    heads: int = 4
    dh: int = 16
    ffn: int = 256
    ctx: int = 64
    act: str = "relu2"          # relu | relu2 | gelu
    quant: bool = True          # False — float-базовая линия той же формы
    tern: bool = True           # при quant: веса тритами (False — только активации целые, для диагностики)
    wq: str = "tern"            # tern: {-1,0,1}*mean|w|;  int2: {-1.5,-0.5,0.5,1.5}*шаг, шаг = 1.25*mean|w|
    wlsq: bool = False          # int3/int4: шаг строки обучается (LSQ), старт — k*mean|w|
    sx: float = 32.0            # шаг int8 на выходе RMSNorm: x8 = round(x * sx)
    score_unit: float = 16.0    # счёт внимания в 1/16 ната — шаг таблицы exp
    score_shift: int = 8        # счёт = (q8 . k8) >> score_shift, в единицах score_unit
    emb_max: int = 63           # вложения в 7 битах: логит = x8 . E8 через таблицу квадратов
    sx_final: float = 16.0      # шаг выхода последней RMSNorm, тоже +-63
    emb_q: str = "int7"         # int7: общий масштаб, +-emb_max;  int4: код -7..7 и обучаемый масштаб строки
    kv_heads: int = 1           # 1 — MQA (K и V одни на все головы); = heads — у каждой головы свои
    prev_emb: bool = False      # к вложению токена прибавляется вложение предыдущего (вторая таблица на входе)
    bigram: int = 0             # строк в таблице пар (предыдущий, текущий) по хешу; 0 — нет таблицы
    experts: int = 1            # MLP — смесь экспертов: каждый токен идёт в одного (вычислений как у одного)
    moe_aux: float = 0.01       # вес потери равномерной загрузки экспертов (как в Switch Transformer)
    pin: int = 0                # первые pin токенов сказки (затравка) всегда в кэше внимания, на любом расстоянии

    def asdict(self):
        return asdict(self)


def cfg_from(d: dict) -> "Cfg":
    """Cfg из чекпойнта; у ранних прогонов не было поздних полей — их старые значения."""
    d = dict(d)
    d.setdefault("emb_max", 127)
    d.setdefault("sx_final", 32.0)
    d.setdefault("wq", "tern")
    d.setdefault("wlsq", False)
    d.setdefault("emb_q", "int7")
    return Cfg(**d)


def ste_round(x):
    return x + (x.round() - x).detach()


def ste_floor(x):
    return x + (x.floor() - x).detach()


def fq_fixed(x, scale, lo, hi):
    """x -> round(x*scale) в [lo, hi], обратно в вещественные. STE."""
    return ste_round((x * scale).clamp(lo, hi)) / scale


class LSQ(nn.Module):
    """Квантование с обучаемым шагом: x ~ s * clamp(round(x/s), lo, hi)."""

    def __init__(self, init_step: float, lo: int, hi: int):
        super().__init__()
        self.log_s = nn.Parameter(torch.tensor(math.log(init_step)))
        self.lo, self.hi = lo, hi

    def forward(self, x, on=True):
        if not on:
            return x
        s = self.log_s.exp()
        return ste_round((x / s).clamp(self.lo, self.hi)) * s

    @property
    def step(self):
        return float(self.log_s.exp())


class TernLinear(nn.Module):
    def __init__(self, din, dout, quant=True, mode="tern", lsq=False):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(dout, din) / math.sqrt(din))
        self.quant = quant
        self.mode = mode
        self.lsq = lsq and mode in ("int3", "int4")
        if self.lsq:
            self.log_step = nn.Parameter(torch.zeros(dout, 1))
            self.register_buffer("step_ready", torch.tensor(0))

    def codes(self, w):
        """(целые коды, масштаб строки): вес = код * масштаб.
        tern: код в {-1,0,1}; int2: код в {-3,-1,1,3}, масштаб — полшага;
        int3: {-3..3}, int4: {-7..7} — шаг под равномерный квантователь для гауссовых весов."""
        a = w.abs().mean(dim=1, keepdim=True).clamp(min=1e-6)
        if self.mode == "tern":
            return (w / a).round().clamp(-1, 1), a
        if self.mode in ("int3", "int4"):
            q, k = {"int3": (3, 0.75), "int4": (7, 0.45)}[self.mode]
            if self.lsq:
                if not bool(self.step_ready):
                    with torch.no_grad():
                        self.log_step.copy_(torch.log(k * a.detach()))
                        self.step_ready.fill_(1)
                step = self.log_step.exp()
                return ste_round((w / step).clamp(-q, q)), step
            step = k * a
            return (w / step).round().clamp(-q, q), step
        step = 1.25 * a
        return (torch.floor(w / step).clamp(-2, 1) * 2 + 1), step / 2

    def ternary(self):
        """(коды int, масштаб строки) — то, что уедет на Спектрум."""
        c, a = self.codes(self.weight.detach())
        return c, a.squeeze(1)

    def forward(self, x):
        w = self.weight
        if self.quant:
            c, a = self.codes(w)
            wq = c * a
            # LSQ: градиент идёт и в веса (через STE внутри codes), и в шаг строки
            w = wq if self.lsq else w + (wq - w).detach()
        return x @ w.t()


def rmsnorm(x, eps=1e-5):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def act_fn(name, x):
    if name == "relu":
        return F.relu(x)
    if name == "relu2":
        return F.relu(x).square()
    if name == "gelu":
        return F.gelu(x)
    raise ValueError(name)


class Block(nn.Module):
    def __init__(self, c: Cfg):
        super().__init__()
        self.c = c
        q = c.quant and c.tern
        L = c.wlsq
        self.wq = TernLinear(c.d, c.heads * c.dh, q, c.wq, L)
        self.wk = TernLinear(c.d, c.kv_heads * c.dh, q, c.wq, L)
        self.wv = TernLinear(c.d, c.kv_heads * c.dh, q, c.wq, L)
        self.wo = TernLinear(c.heads * c.dh, c.d, q, c.wq, L)
        self.w1 = TernLinear(c.d, c.ffn, q, c.wq, L)
        self.w2 = TernLinear(c.ffn, c.d, q, c.wq, L)
        if c.experts > 1:
            # эксперт 0 — w1/w2 выше (старт из плотной модели), остальные — свои пары матриц
            self.ew1 = nn.ModuleList([TernLinear(c.d, c.ffn, q, c.wq, L) for _ in range(c.experts - 1)])
            self.ew2 = nn.ModuleList([TernLinear(c.ffn, c.d, q, c.wq, L) for _ in range(c.experts - 1)])
            self.router = TernLinear(c.d, c.experts, q, c.wq, L)
            self.aux = None
        # усиление q по головам: заменяет 1/sqrt(dh) и подгоняет q под общий сдвиг счёта
        self.q_gain = nn.Parameter(torch.full((c.heads,), 3.0 if c.quant else 1.0))
        self.fk = LSQ(0.05, -63, 63)
        self.fv = LSQ(0.05, -63, 63)
        self.fpre = LSQ(0.05, -127, 127)
        self.fpost = LSQ(0.05, 0 if c.act != "gelu" else -127, 127)
        # относительная позиция: добавка к счёту по расстоянию 0..ctx-1, начало — наклоны ALiBi
        slopes = torch.tensor([2.0 ** (-8.0 * (h + 1) / c.heads) for h in range(c.heads)])
        dist = torch.arange(c.ctx).float()
        self.rel_bias = nn.Parameter(-slopes[:, None] * dist[None, :] * 4.0)
        if c.pin:
            # добавка к счёту для закреплённых токенов начала (одна на голову, расстояние не важно)
            self.pin_bias = nn.Parameter(self.rel_bias.detach()[:, -1].clone())

    def x8(self, h):
        x = rmsnorm(h)
        return fq_fixed(x, self.c.sx, -127, 127) if self.c.quant else x

    def attention(self, h, P=0, amask=None):
        """h: (B, P+T, d) — сначала P закреплённых токенов начала сказки, потом окно. amask (B, T, P) —
        какие закреплённые видит каждая позиция окна (не из будущего и из той же сказки)."""
        c = self.c
        B, T, _ = h.shape
        x = self.x8(h)
        q = self.wq(x).view(B, T, c.heads, c.dh)
        g = c.heads // c.kv_heads                        # голов на одну пару K/V
        k = self.wk(x).view(B, T, c.kv_heads, c.dh).repeat_interleave(g, 2)     # B T H dh
        v = self.wv(x).view(B, T, c.kv_heads, c.dh).repeat_interleave(g, 2)
        idx = torch.arange(T, device=h.device)
        dist = (idx[:, None] - idx[None, :])             # t - s
        causal = dist >= 0
        rb = ste_round(self.rel_bias) if c.quant else self.rel_bias
        bias = rb[:, dist.clamp(0, c.ctx - 1)]           # H T T
        if P:
            pb = ste_round(self.pin_bias) if c.quant else self.pin_bias
            bias = torch.cat([bias[:, :, :P] * 0 + pb[:, None, None], bias[:, :, P:]], 2)
            bias = torch.cat([rb[:, dist[:P].clamp(0, c.ctx - 1)], bias[:, P:]], 1)
            causal = causal[None].repeat(B, 1, 1)
            causal[:, P:, :P] = amask
            causal = causal[:, None]                     # B 1 T T
        if c.quant:
            # Как на Z80: q8, k8, v8 в 7 битах (+-63) — умножение по таблице четвертей квадратов
            # без 9-битных сумм. Счёт — целый, в 1/score_unit ната: (q8.k8 >> shift) + добавка
            # по расстоянию. Вес позиции e = round(63*exp((s-max)/unit)) — 6 бит от максимума,
            # выход внимания o = round(8 * sum(e*v8) / sum(e)) — в шагах v/8.
            k8 = self.fk(k) / self.fk.log_s.exp()
            q8 = ste_round((q * self.q_gain.view(1, 1, -1, 1) * 16.0).clamp(-63, 63))
            raw = torch.einsum("bthd,bshd->bhts", q8, k8)
            s = ste_floor(raw / (2 ** c.score_shift)) + bias[None]
            s = s.masked_fill(~causal, float("-inf"))
            m = s.max(-1, keepdim=True).values.detach()
            e = ste_round(63.0 * torch.exp((s - m) / c.score_unit))
            sv = self.fv.log_s.exp()
            v8 = self.fv(v) / sv
            o = torch.einsum("bhts,bshd->bthd", e, v8) / e.sum(-1).transpose(1, 2).unsqueeze(-1)
            o = ste_round(o * 8.0) * (sv / 8.0)
            return self.wo(o.reshape(B, T, c.heads * c.dh))
        scores = torch.einsum("bthd,bshd->bhts", q * self.q_gain.view(1, 1, -1, 1), k) / math.sqrt(c.dh)
        scores = scores + bias[None] / c.score_unit
        scores = scores.masked_fill(~causal, float("-inf"))
        p = scores.softmax(-1)
        o = torch.einsum("bhts,bshd->bthd", p, v).reshape(B, T, c.heads * c.dh)
        return self.wo(o)

    def expert(self, x, w1, w2):
        c = self.c
        u = w1(x)
        if c.quant and c.act != "relu":
            u = self.fpre(u)
        u = act_fn(c.act, u)
        u = self.fpost(u, c.quant)
        return w2(u)

    def mlp(self, h):
        c = self.c
        x = self.x8(h)
        if c.experts == 1:
            return self.expert(x, self.w1, self.w2)
        # жёсткий выбор одного эксперта (как на Z80); градиент в маршрутизатор — через p/p.detach() = 1
        r = self.router(x).float()
        p = r.softmax(-1)
        top = r.argmax(-1)                                            # B T
        pt = p.gather(-1, top[..., None])
        outs = [self.expert(x, self.w1, self.w2)] + [self.expert(x, a, b) for a, b in zip(self.ew1, self.ew2)]
        y = torch.stack(outs, -2).gather(-2, top[..., None, None].expand(*top.shape, 1, c.d)).squeeze(-2)
        f = torch.nn.functional.one_hot(top, c.experts).float().mean((0, 1))
        self.aux = c.experts * (f * p.mean((0, 1))).sum()
        return y * (pt / pt.detach())

    def forward(self, h, P=0, amask=None):
        h = h + self.attention(h, P, amask)
        h = h + self.mlp(h)
        return h


class TinyGPT(nn.Module):
    def __init__(self, c: Cfg):
        super().__init__()
        self.c = c
        self.emb = nn.Parameter(torch.randn(c.vocab, c.d) * 0.5)
        if c.emb_q == "int4":
            self.emb_log_step = nn.Parameter(torch.zeros(c.vocab, 1))
            self.register_buffer("emb_step_ready", torch.tensor(0))
        self.blocks = nn.ModuleList([Block(c) for _ in range(c.layers)])
        self.logit_gain = nn.Parameter(torch.tensor(1.0))
        # входные таблицы без вычислений: предыдущий токен и пара (предыдущий, текущий).
        # Квантуются, как вложения int4: код -7..7 и обучаемый масштаб строки.
        if c.prev_emb:
            self.prev = nn.Parameter(torch.randn(c.vocab, c.d) * 0.1)
            self.prev_log_step = nn.Parameter(torch.zeros(c.vocab, 1))
            self.register_buffer("prev_step_ready", torch.tensor(0))
        if c.bigram:
            assert c.bigram & (c.bigram - 1) == 0
            self.big = nn.Parameter(torch.randn(c.bigram, c.d) * 0.1)
            self.big_log_step = nn.Parameter(torch.zeros(c.bigram, 1))
            self.register_buffer("big_step_ready", torch.tensor(0))
            # табличный хеш: строка = (H1[предыдущий] ^ H2[текущий]) & (bigram-1) — на Z80 два чтения и XOR
            g = torch.Generator().manual_seed(12345)
            self.register_buffer("h1", torch.randint(0, 1 << 15, (c.vocab,), generator=g))
            self.register_buffer("h2", torch.randint(0, 1 << 15, (c.vocab,), generator=g))

    def emb_codes(self):
        """int4: (коды -7..7, масштаб строки)."""
        e = self.emb
        if not bool(self.emb_step_ready):
            with torch.no_grad():
                self.emb_log_step.copy_(torch.log(0.45 * e.abs().mean(dim=1, keepdim=True)))
                self.emb_step_ready.fill_(1)
        step = self.emb_log_step.exp()
        return ste_round((e / step).clamp(-7, 7)), step

    def emb_q(self):
        e = self.emb
        if not self.c.quant:
            return e
        if self.c.emb_q == "int4":
            c, step = self.emb_codes()
            return c * step
        q = self.c.emb_max
        s = e.detach().abs().max() / q
        return ste_round((e / s).clamp(-q, q)) * s

    def table_q(self, w, log_step, ready):
        """Входная таблица int4 (при quant): код -7..7 и обучаемый масштаб строки."""
        if not (self.c.quant and self.c.emb_q == "int4"):
            return w
        if not bool(ready):
            with torch.no_grad():
                log_step.copy_(torch.log(0.45 * w.abs().mean(dim=1, keepdim=True)))
                ready.fill_(1)
        step = log_step.exp()
        return ste_round((w / step).clamp(-7, 7)) * step

    def bigram_row(self, prev, idx):
        return (self.h1[prev] ^ self.h2[idx]) & (self.c.bigram - 1)

    def embed(self, idx, prev0=None):
        """Вход остаточного потока: вложение токена + входные таблицы предыдущего токена и пары.
        prev0 (B,1) — токен перед окном; нет — конец сказки."""
        c = self.c
        h = self.emb_q()[idx]
        if c.prev_emb or c.bigram:
            p0 = prev0 if prev0 is not None else torch.zeros_like(idx[:, :1])
            prev = torch.cat([p0, idx[:, :-1]], 1)
            if c.prev_emb:
                h = h + self.table_q(self.prev, self.prev_log_step, self.prev_step_ready)[prev]
            if c.bigram:
                h = h + self.table_q(self.big, self.big_log_step, self.big_step_ready)[self.bigram_row(prev, idx)]
        return h

    def forward(self, idx, targets=None, prev0=None, pin=None, amask=None):
        """pin (B, P) — первые токены сказки окна (начиная с её конца-начала), amask (B, T, P) — см. attention."""
        c = self.c
        e = self.emb_q()
        h = self.embed(idx, prev0)
        P = 0
        if c.pin and pin is not None:
            P = pin.shape[1]
            h = torch.cat([self.embed(pin), h], 1)
        for b in self.blocks:
            h = b(h, P, amask)
        h = h[:, P:]
        x = rmsnorm(h)
        if c.quant:
            lim = 63 if c.sx_final <= 16 else 127
            x = fq_fixed(x, c.sx_final, -lim, lim)
        logits = (x @ e.t()) * self.logit_gain
        if targets is None:
            return logits
        loss = F.cross_entropy(logits.reshape(-1, c.vocab), targets.reshape(-1))
        if c.experts > 1 and self.training:
            # потеря загрузки — только в градиент; возвращаемая потеря остаётся чистой CE для отчёта
            aux = sum(b.aux for b in self.blocks) * c.moe_aux
            loss = loss + aux - aux.detach()
        return logits, loss


def count_params(c: Cfg):
    per_layer = c.d * c.heads * c.dh * 2 + c.d * c.dh * 2 + c.d * c.ffn * 2
    return dict(tern=per_layer * c.layers, emb=c.vocab * c.d,
                bytes_2bit=per_layer * c.layers // 4, bytes_b3=math.ceil(per_layer * c.layers / 5),
                kv_bytes=c.layers * c.ctx * c.dh * 2)
