"""Lane G / G2: cross-encoder TRAINING kernel on Kaggle (2x T4 or 1x L4). Self-contained (no `ber` import).

Inputs: private dataset from pipelines/xenc_g1_export.py (train.parquet / valid.parquet: pair_id, text_a, text_b,
label, cf). Cross-fit: a model trained on cf == k never sees S1 of half 1-k -> its logits on half 1-k are OOF.
Models (MIT): FacebookAI/xlm-roberta-base (278M) | microsoft/Multilingual-MiniLM-L12-H384 (117M, XLM-R tokenizer).

CONFIG (edited per job by build step; see kaggle/README_lane_g.md):
  mode "dp":   ONE model on half HALVES[0], DataParallel over all GPUs (XLM-R: acc2 half 0, acc3 half 1)
  mode "pair": one model per half, one per GPU in parallel (MiniLM both halves on acc5)
Recipe (lead 19:45 + 20:10): BCE, fp16 AMP, max_len 96, 1 epoch, warmup 5%, linear decay, time-boxed: after a
200-step throughput probe the epoch is truncated to fit BUDGET_MIN (schedule follows the truncated length).
Checkpoint + val logloss every 30 min; the best checkpoint (by val logloss) is kept as ckpt_best_h{k}.pt.
Augmentation (train positives only, p = 0.15, generic / no country logic): name-token drop, adjacent-token swap,
legal-suffix drop. No digit edits (never create fake house-number positives).
Outputs (/kaggle/working): ckpt_best_h{k}.pt, ckpt_last_h{k}.pt (fp16 state_dict), progress_h{k}.json, metrics.json.
"""
import glob
import json
import math
import os
import random
import re
import time

import numpy as np
import pandas as pd
import torch

CONFIG = {"model": "microsoft/Multilingual-MiniLM-L12-H384", "mode": "pair", "halves": [0, 1], "lr": 5e-05, "budget_min": 170, "bs_per_gpu": 128}  # @CONFIG@ (generated from xenc_g2.py)
MAX_LEN, EVAL_BS, WD, WARM, SEED, P_AUG = 96, 512, 0.01, 0.05, 42, 0.15
CKPT_EVERY_S, VAL_N, PROBE_STEPS = 30 * 60, 60_000, 200
W, IN = "/kaggle/working", "/kaggle/input"
LEGAL = {"inc", "llc", "ltd", "corp", "co", "pvt", "llp", "plc", "sarl", "sas", "sasu", "sa", "eurl", "sci", "snc",
         "gmbh", "pc", "pllc", "lp"}


def log(k, msg):
    print(f"[{time.strftime('%H:%M:%S')}] h{k} {msg}", flush=True)


def find(name):
    fs = glob.glob(f"{IN}/**/{name}", recursive=True)
    assert fs, f"missing input {name}"
    return fs[0]


def tokenizer():
    from transformers import AutoTokenizer, XLMRobertaTokenizerFast
    if "MiniLM" in CONFIG["model"]:   # MiniLM ships only the slow spm; xlm-roberta-base's fast tokenizer = same ids
        return XLMRobertaTokenizerFast.from_pretrained("FacebookAI/xlm-roberta-base")
    return AutoTokenizer.from_pretrained(CONFIG["model"])


def load_model():
    from transformers import AutoModelForSequenceClassification, BertForSequenceClassification
    if "MiniLM" in CONFIG["model"]:
        return BertForSequenceClassification.from_pretrained(CONFIG["model"], num_labels=1)
    return AutoModelForSequenceClassification.from_pretrained(CONFIG["model"], num_labels=1)


_NAME = re.compile(r"^(\[COL\] name \[VAL\] )(.*?)( \[COL\] address .*)$")


def augment(text, rng):
    m = _NAME.match(text)
    if not m:
        return text
    toks = m.group(2).split()
    if len(toks) < 2:
        return text
    r = rng.random()
    if r < 1 / 3:
        toks.pop(rng.randrange(1, len(toks)))                                     # token drop (keep the first)
    elif r < 2 / 3:
        i = rng.randrange(len(toks) - 1)
        toks[i], toks[i + 1] = toks[i + 1], toks[i]                               # adjacent swap
    else:
        kept = [t for t in toks if t not in LEGAL]                                # legal-suffix drop
        toks = kept if kept and len(kept) < len(toks) else toks
    return m.group(1) + " ".join(toks) + m.group(3)


def encode(tok, a, b, bs=20_000):
    ids = []
    for o in range(0, len(a), bs):
        enc = tok(a[o:o + bs], b[o:o + bs], truncation="longest_first", max_length=MAX_LEN,
                  return_attention_mask=False, return_token_type_ids=False)
        ids.extend(np.asarray(x, dtype=np.int32) for x in enc["input_ids"])
    return ids


def collate(seqs, pad_id, device):
    L = max(len(s) for s in seqs)
    x = np.full((len(seqs), L), pad_id, dtype=np.int64)
    for i, s in enumerate(seqs):
        x[i, :len(s)] = s
    x = torch.from_numpy(x).to(device, non_blocking=True)
    return x, (x != pad_id).long()


@torch.inference_mode()
def predict(model, ids, pad_id, device):
    was = model.training
    model.eval()
    order = np.argsort([len(s) for s in ids], kind="stable")
    out = np.empty(len(ids), dtype=np.float32)
    for o in range(0, len(ids), EVAL_BS):
        b = order[o:o + EVAL_BS]
        x, m = collate([ids[i] for i in b], pad_id, device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            out[b] = model(input_ids=x, attention_mask=m).logits.float().squeeze(-1).cpu().numpy()
    model.train(was)
    return out


def logloss(y, lg):
    p = 1 / (1 + np.exp(-np.clip(lg, -30, 30)))
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def train_half(k, devices):
    """Train one model on half k. devices: list of cuda ids (len > 1 -> DataParallel)."""
    from transformers import get_linear_schedule_with_warmup  # noqa: F401  (import check only)
    dev = torch.device(f"cuda:{devices[0]}")
    torch.cuda.set_device(dev)
    torch.manual_seed(SEED + k)
    rng, prng = random.Random(SEED + k), np.random.default_rng(SEED + k)
    tok = tokenizer()
    pad = tok.pad_token_id
    net = load_model().to(dev)
    model = torch.nn.DataParallel(net, device_ids=devices) if len(devices) > 1 else net
    tr = pd.read_parquet(find("train.parquet"))
    tr = tr[tr.cf == k].reset_index(drop=True)
    va = pd.read_parquet(find("valid.parquet")).sample(VAL_N, random_state=SEED)
    va_ids, va_y = encode(tok, va.text_a.tolist(), va.text_b.tolist()), va.label.to_numpy(np.float32)
    tb = tr.text_b.tolist()
    y = tr.label.to_numpy(np.float32)
    n_aug = 0
    for i in np.flatnonzero(y == 1):
        if rng.random() < P_AUG:
            tb[i] = augment(tb[i], rng)
            n_aug += 1
    t0 = time.time()
    ids = encode(tok, tr.text_a.tolist(), tb)
    log(k, f"train rows {len(tr):,} pos {y.mean():.4f} aug {n_aug:,}; tokenized {time.time() - t0:.0f}s, "
           f"mean len {np.mean([len(s) for s in ids]):.1f}; devices {devices}")
    del tr, tb
    bs = CONFIG["bs_per_gpu"] * len(devices)
    steps = math.ceil(len(ids) / bs)
    plan = {"total": steps}
    warm = max(1, int(WARM * steps))
    opt = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=WD)
    sch = torch.optim.lr_scheduler.LambdaLR(      # warmup, then linear decay to the (possibly truncated) plan
        opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (plan["total"] - s) / max(1, plan["total"] - warm)))
    scaler = torch.amp.GradScaler("cuda")
    lossf = torch.nn.BCEWithLogitsLoss()
    perm = prng.permutation(len(ids))
    prog = {"half": k, "model": CONFIG["model"], "devices": [torch.cuda.get_device_name(d) for d in devices],
            "rows": len(ids), "steps_full_epoch": steps, "bs": bs}
    best, t_start, t_ck, run_loss = float("inf"), time.time(), time.time(), 0.0
    budget_s = CONFIG["budget_min"] * 60
    model.train()
    s = 0
    while s < plan["total"]:
        b = perm[s * bs:(s + 1) * bs]
        x, m = collate([ids[i] for i in b], pad, dev)
        yt = torch.from_numpy(y[b]).to(dev)
        with torch.autocast("cuda", dtype=torch.float16):
            logit = model(input_ids=x, attention_mask=m).logits.squeeze(-1)
        loss = lossf(logit.float(), yt)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sch.step()
        run_loss = 0.98 * run_loss + 0.02 * loss.item() if s else loss.item()
        s += 1
        if s == PROBE_STEPS:
            torch.cuda.synchronize()
            rate = s * bs / (time.time() - t_start)
            fit = int(rate * (budget_s - (time.time() - t_start)) / bs) + s
            plan["total"] = min(steps, fit)
            prog.update(train_pairs_per_s=round(rate), planned_steps=plan["total"],
                        epoch_frac=round(plan["total"] / steps, 3), eta_min=round((plan["total"] - s) * bs / rate / 60, 1))
            log(k, f"PROBE {s} steps: {rate:.0f} pairs/s -> {plan['total']}/{steps} steps "
                   f"({plan['total'] / steps:.0%} of the epoch) in the {CONFIG['budget_min']} min budget; loss {run_loss:.4f}")
            json.dump(prog, open(f"{W}/progress_h{k}.json", "w"), indent=2)
        if s % 500 == 0:
            log(k, f"step {s}/{plan['total']} loss {run_loss:.4f} lr {sch.get_last_lr()[0]:.2e}")
        if time.time() - t_ck > CKPT_EVERY_S or s == plan["total"]:
            t_ck = time.time()
            vl = logloss(va_y, predict(model, va_ids, pad, dev))
            sd = {n: t.half() for n, t in net.state_dict().items()}
            torch.save(sd, f"{W}/ckpt_last_h{k}.pt")
            if vl < best:
                best = vl
                torch.save(sd, f"{W}/ckpt_best_h{k}.pt")
            prog.setdefault("ckpts", []).append({"step": s, "min": round((time.time() - t_start) / 60, 1),
                                                 "train_loss": round(run_loss, 5), "val_logloss": round(vl, 5)})
            log(k, f"ckpt step {s}: val logloss {vl:.5f} (best {best:.5f})")
            json.dump(prog, open(f"{W}/progress_h{k}.json", "w"), indent=2)
    prog.update(train_min=round((time.time() - t_start) / 60, 1), best_val_logloss=round(best, 5))
    json.dump(prog, open(f"{W}/progress_h{k}.json", "w"), indent=2)
    log(k, f"done: {prog['train_min']} min, best val logloss {best:.5f}")


def _pair_worker(i, halves):
    train_half(halves[i], [i])


if __name__ == "__main__":
    import torch.multiprocessing as mp
    t0 = time.time()
    n_gpu = torch.cuda.device_count()
    print(f"CONFIG {json.dumps(CONFIG)}; GPUs {[torch.cuda.get_device_name(i) for i in range(n_gpu)]}", flush=True)
    assert n_gpu >= 1, "no GPU"
    tokenizer()
    load_model()                                        # warm the HF cache before workers start
    halves = CONFIG["halves"]
    if CONFIG["mode"] == "pair" and n_gpu >= 2 and len(halves) == 2:
        try:
            mp.spawn(_pair_worker, args=(halves,), nprocs=2, join=True)
        except Exception as e:
            import threading
            print(f"mp.spawn failed ({type(e).__name__}: {e}); using 1 thread per GPU", flush=True)
            th = [threading.Thread(target=train_half, args=(h, [i])) for i, h in enumerate(halves)]
            [t.start() for t in th]; [t.join() for t in th]
    elif CONFIG["mode"] == "pair":
        for h in halves:                                 # 1 GPU: sequential, budget applies per half
            train_half(h, [0])
    else:
        train_half(halves[0], list(range(n_gpu)))
    m = {os.path.basename(p): json.load(open(p)) for p in sorted(glob.glob(f"{W}/progress_h*.json"))}
    json.dump(m, open(f"{W}/metrics.json", "w"), indent=2)
    print(f"total {(time.time() - t0) / 60:.1f} min", flush=True)
