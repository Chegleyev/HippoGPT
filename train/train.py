"""Обучение TinyGPT на потоке символов TinyStories.

Пример:  python train/train.py --name base --steps 20000
Метрика — биты на символ (bpc) на фиксированном куске valid, одном для всех прогонов.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from model import Cfg, TinyGPT, count_params

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"


def load(name):
    """Корпус остаётся в ОЗУ (memmap, общий кэш страниц на все процессы), на GPU — только батч.
    *.u8 — символы, *.u16 — токены BPE."""
    dt = np.uint16 if name.endswith(".u16") else np.uint8
    return np.memmap(ROOT / "data" / name, dtype=dt, mode="r")


_EOS = {}


def eos_index(data):
    """Позиции концов сказок (токен 0) — по ним находится начало сказки окна для закреплённых токенов."""
    key = (id(data), len(data))
    if key not in _EOS:
        fn = getattr(data, "filename", None)
        cache = Path(str(fn) + ".eos.npy") if fn else None
        if cache is not None and cache.exists():
            _EOS[key] = np.load(cache)
        else:
            idx = np.concatenate([np.flatnonzero(np.asarray(data[i:i + 50_000_000]) == 0) + i
                                  for i in range(0, len(data), 50_000_000)]).astype(np.int64)
            if cache is not None:
                np.save(cache, idx)
            _EOS[key] = idx
    return _EOS[key]


def pin_inputs(data, ix, T, P, x, device):
    """Окно с входами data[ix+1 .. ix+T] (x — они же на устройстве): закреплённые первые P токенов его сказки
    (с её начального конца-сказки) и маска (B, T, P): позиция видит закреплённый j, если j не дальше её места
    в сказке и в окне до неё не началась новая сказка."""
    eos = eos_index(data)
    k = np.searchsorted(eos, ix + 1, side="right") - 1
    ss = np.where(k >= 0, eos[np.maximum(k, 0)], 0)
    pin = np.asarray(data[np.minimum(ss[:, None] + np.arange(P)[None], len(data) - 1)]).astype(np.int64)
    sp = torch.from_numpy((ix + 1 - ss)[:, None] + np.arange(T)[None]).to(device)        # место в сказке
    newst = torch.cumsum((x[:, 1:] == 0).long(), 1) > 0
    newst = torch.cat([torch.zeros_like(newst[:, :1]), newst], 1)
    am = (torch.arange(P, device=device)[None, None] <= sp[:, :, None]) & ~newst[:, :, None]
    return torch.from_numpy(pin).to(device), am


def batches(data, B, T, seed, device, P=0):
    """(вход, цель, токен перед окном, закреплённые, маска) — токен перед окном нужен входным таблицам,
    закреплённые (P > 0) — первые токены сказки окна."""
    rng = np.random.default_rng(seed)
    n = len(data) - T - 2
    ar = np.arange(T + 2)[None]
    while True:
        ix = rng.integers(0, n, B)
        chunk = torch.from_numpy(np.asarray(data[ix[:, None] + ar]).astype(np.int64)).to(device, non_blocking=True)
        x = chunk[:, 1:-1]
        pin, am = pin_inputs(data, ix, T, P, x, device) if P else (None, None)
        yield x, chunk[:, 2:], chunk[:, :1], pin, am


@torch.no_grad()
def evaluate(model, val, T, n_windows=4096, B=512, piece_len=None, val_np=None):
    """Биты на символ на первых n_windows неперекрывающихся окнах valid (всегда одних и тех же).
    Для токенов: сумма потерь по предсказанным токенам / сумма длин их строк."""
    model.eval()
    P = model.c.pin
    tot, chars = 0.0, 0
    for i in range(0, n_windows, B):
        starts = torch.arange(i, min(i + B, n_windows), device=val.device) * T
        off = starts[:, None] + torch.arange(T + 1, device=val.device)[None]
        chunk = val[off].long()
        p0 = val[(starts - 1).clamp(min=0)].long()[:, None] * (starts > 0).long()[:, None]
        pin, am = (pin_inputs(val_np, starts.cpu().numpy() - 1, T, P, chunk[:, :-1], val.device) if P
                   else (None, None))
        _, loss = model(chunk[:, :-1], chunk[:, 1:], p0, pin, am)
        n = chunk[:, 1:].numel()
        tot += float(loss) * n
        chars += int(piece_len[chunk[:, 1:]].sum()) if piece_len is not None else n
    model.train()
    return tot / chars / math.log(2)


@torch.no_grad()
def calibrate_from_float(model, data, cfg, dev):
    """Старт QAT из float-модели: шаги LSQ — по 99.9-му перцентилю модулей на калибровочном
    батче (прогон в float), q_gain — так, чтобы целочисленный счёт внимания совпал с float."""
    import model as M
    lins = [m for m in model.modules() if isinstance(m, M.TernLinear)]
    cfg.quant = False
    for m in lins:
        m.quant = False
    x = next(batches(data, 256, cfg.ctx, 123, dev))[0]
    h = model.emb[x]
    pct = lambda t, Q: float(torch.quantile(t.abs().flatten()[:500000].float(), 0.999)) / Q
    stats = []
    for b in model.blocks:
        xx = M.rmsnorm(h)
        sk, sv = pct(b.wk(xx), 63), pct(b.wv(xx), 63)
        h1 = h + b.attention(h)
        u = b.w1(M.rmsnorm(h1))
        spre, spost = pct(u, 127), pct(M.act_fn(cfg.act, u), 127)
        h = h1 + b.mlp(h1)
        stats.append((sk, sv, spre, spost))
    cfg.quant = True
    for m in lins:
        m.quant = cfg.tern
    for b, (sk, sv, spre, spost) in zip(model.blocks, stats):
        b.fk.log_s.fill_(math.log(sk))
        b.fv.log_s.fill_(math.log(sv))
        b.fpre.log_s.fill_(math.log(spre))
        b.fpost.log_s.fill_(math.log(spost))
        # float: q.k*g/sqrt(dh);  QAT: (q*g'*16).(k/sk)/2^shift/unit  ->  g' = g*2^shift*unit*sk/(16*sqrt(dh))
        b.q_gain.mul_(2 ** cfg.score_shift * cfg.score_unit * sk / (16 * math.sqrt(cfg.dh)))
    print("калибровка из float:", [(round(b.fk.step, 4), round(b.fv.step, 4),
                                   [round(float(g), 2) for g in b.q_gain]) for b in model.blocks], flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--init", type=str, default=None, help="чекпойнт для старта (float -> QAT)")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--teacher", type=str, default=None, help="чекпойнт учителя для дистилляции")
    ap.add_argument("--data", type=str, default="", help="tok256 / tok512 — корпус в токенах BPE")
    ap.add_argument("--mix", type=str, default="", help="имя:доля — подмешать корпус data/имя_train.u16 "
                    "(например speccy512:0.1), и считать bpc на data/имя_valid.u16")
    ap.add_argument("--train-data", type=str, default="", help="учиться на data/<имя>_train.u16 (тот же словарь), "
                    "проверять — на обычном valid из --data")
    ap.add_argument("--kd", type=float, default=0.9, help="доля KL к учителю в потере (остальное — CE по тексту)")
    for k, v in Cfg().asdict().items():
        ap.add_argument(f"--{k.replace('_', '-')}", type=type(v) if not isinstance(v, bool) else int, default=v)
    a = ap.parse_args()

    cfg = Cfg(**{k: (bool(getattr(a, k)) if isinstance(v, bool) else getattr(a, k))
                 for k, v in Cfg().asdict().items()})
    torch.manual_seed(a.seed)
    dev = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    out = RUNS / a.name
    out.mkdir(parents=True, exist_ok=True)

    piece_len = None
    if a.data:
        meta = json.loads((ROOT / "data" / f"{a.data}.json").read_text(encoding="utf-8"))
        assert len(meta["pieces"]) == cfg.vocab, "--vocab должен совпадать со словарём"
        piece_len = torch.tensor([len(p) for p in meta["pieces"]], device=dev)
        train = load(f"{a.train_data or a.data}_train.u16")
        val_np = np.array(load(f"{a.data}_valid.u16")[:4096 * cfg.ctx + 1])
        val = torch.from_numpy(val_np.astype(np.int64)).to(dev)
    else:
        train = load("train.u8")
        val_np = np.array(load("valid.u8")[:4096 * cfg.ctx + 1])
        val = torch.from_numpy(val_np).to(dev)
    model = TinyGPT(cfg).to(dev)
    if a.init:
        src = torch.load(a.init, map_location=dev)
        model.load_state_dict(src["model"], strict=False)
        if not src["cfg"].get("quant", True) and cfg.quant:
            calibrate_from_float(model, train, cfg, dev)
    print(cfg, count_params(cfg), flush=True)
    teacher = None
    if a.teacher:
        from model import cfg_from
        tsrc = torch.load(a.teacher, map_location=dev)
        teacher = TinyGPT(cfg_from(tsrc["cfg"])).to(dev)
        teacher.load_state_dict(tsrc["model"])
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        print("учитель", a.teacher, "val_bpc", round(tsrc["val_bpc"], 4), "kd", a.kd, flush=True)

    decay, no_decay = [], []
    for n, p in model.named_parameters():
        (decay if p.dim() == 2 and "rel_bias" not in n and "log_step" not in n else no_decay).append(p)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.wd},
                             {"params": no_decay, "weight_decay": 0.0}], lr=a.lr, betas=(0.9, 0.95))
    step_fn = model
    if a.compile:
        step_fn = torch.compile(model)

    it = batches(train, a.batch, cfg.ctx, a.seed, dev, cfg.pin)
    val2 = None
    if a.mix:
        mname, mfrac = a.mix.split(":")
        b2 = max(1, round(a.batch * float(mfrac)))
        it1 = batches(train, a.batch - b2, cfg.ctx, a.seed, dev, cfg.pin)
        it2 = batches(load(f"{mname}_train.u16"), b2, cfg.ctx, a.seed + 1, dev, cfg.pin)
        cat = lambda u, w: None if u is None else torch.cat([u, w])
        it = (tuple(cat(u, w) for u, w in zip(b1, b2_)) for b1, b2_ in zip(it1, it2))
        v2 = np.array(load(f"{mname}_valid.u16"))
        v2_np = v2
        n2 = (len(v2) - 1) // cfg.ctx
        val2 = (torch.from_numpy(v2[:n2 * cfg.ctx + 1].astype(np.int64)).to(dev), n2)
        print(f"подмешиваю {mname}: {b2} из {a.batch} строк батча, проверка — {n2} окон", flush=True)
    log = []
    t0 = time.time()
    best = 9.9
    for step in range(1, a.steps + 1):
        lr = a.lr * min(1.0, step / a.warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / a.steps)))
        for g in opt.param_groups:
            g["lr"] = lr
        x, y, p0, pin, am = next(it)
        logits, loss = step_fn(x, y, p0, pin, am)
        if teacher is not None:
            # дистилляция внутри QAT: ученик всё время на квантованных весах, цель — распределение учителя
            with torch.no_grad():
                t_logp = torch.log_softmax(teacher(x, prev0=p0).float(), -1)
            s_logp = torch.log_softmax(logits.float(), -1)
            kl = torch.nn.functional.kl_div(s_logp.reshape(-1, cfg.vocab), t_logp.reshape(-1, cfg.vocab),
                                            log_target=True, reduction="batchmean")
            loss_total = a.kd * kl + (1 - a.kd) * loss
        else:
            loss_total = loss
        opt.zero_grad(set_to_none=True)
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % a.eval_every == 0 or step == a.steps:
            bpc = evaluate(model, val, cfg.ctx, piece_len=piece_len, val_np=val_np)
            el = time.time() - t0
            chars = step * a.batch * cfg.ctx
            rec = dict(step=step, train_bpc=float(loss.detach()) / math.log(2), val_bpc=bpc, sec=el, mchars=chars / 1e6)
            if val2 is not None:
                rec["val2_bpc"] = evaluate(model, val2[0], cfg.ctx, n_windows=val2[1], piece_len=piece_len,
                                           val_np=v2_np)
            log.append(rec)
            print(json.dumps(rec), flush=True)
            if bpc < best:
                best = bpc
            torch.save({"model": model.state_dict(), "cfg": cfg.asdict(), "step": step, "val_bpc": bpc,
                        "data": a.data},
                       out / "last.pt")
    (out / "log.json").write_text(json.dumps(dict(cfg=cfg.asdict(), args=vars(a), log=log), indent=1))
    print("done", a.name, "val_bpc", log[-1]["val_bpc"])


if __name__ == "__main__":
    main()
