"""Cross-encoder (Stage 5) on Kaggle 2x T4. Self-contained (no `ber` import).

Inputs (private dataset amlc-xenc-data, from pipelines/xenc_export.py): *.parquet with pair_id, text_a, text_b
[, label], cf.  Cross-fit: model k (one per GPU, in parallel) trains on train rows with cf == k, then scores
rows with cf == 1-k (out-of-fold) and cf == -1 (valid / eval / test: both models).

Model: microsoft/Multilingual-MiniLM-L12-H384 (MIT, 117M params), BertForSequenceClassification(num_labels=1),
BCE, max_len 96, fp16 AMP, 1 epoch, AdamW lr 5e-5 with linear warmup/decay, seed 42.

Outputs (/kaggle/working): ckpt_m{k}.pt (fp16 state_dict, written progressively), progress_m{k}.json,
logits_<name>.parquet (pair_id, xenc_l0, xenc_l1, xenc_logit), metrics.json.
LOCO mode: if a loco_groups.parquet (pair_id, grp) + loco_config.json ({"train_grp": ["US", "India"]}) are in the
inputs, model k trains on ALL train rows with grp == train_grp[k] (country used only to SPLIT, never as a feature),
scores every non-train row, and outputs are prefixed "loco_" (xenc_l0 = model 0, xenc_l1 = model 1).
Score-only rerun: attach this kernel's output as a kernel source; found ckpt_m{k}.pt are loaded (no training) and
files whose logits_<name>.parquet already exist in the inputs are skipped.
"""
import glob
import json
import math
import os
import time

import numpy as np
import pandas as pd
import torch

MODEL = "microsoft/Multilingual-MiniLM-L12-H384"
TOKENIZER = "FacebookAI/xlm-roberta-base"         # tokenizer files only (MIT); see tokenizer()
MAX_LEN, BS, EVAL_BS, LR, WD, WARM, SEED = 96, 128, 1024, 5e-5, 0.01, 0.06, 42
PROBE_FRAC = 0.01
N_CKPT = 4                                           # checkpoints per epoch
W = os.environ.get("XENC_OUT", "/kaggle/working")
IN = os.environ.get("XENC_IN", "/kaggle/input")
SMOKE = int(os.environ.get("XENC_SMOKE", "0"))              # >0: CPU smoke test on the first SMOKE rows per file
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")


def log(k, msg):
    print(f"[{time.strftime('%H:%M:%S')}] m{k} {msg}", flush=True)


def inputs(pattern):
    return sorted(glob.glob(f"{IN}/**/{pattern}", recursive=True))


def data_files():
    fs = [f for f in inputs("*.parquet") if not os.path.basename(f).startswith(("logits_", "part_"))]
    return {os.path.basename(f)[:-8]: f for f in fs}


def tokenizer():
    # MiniLM ships only the slow XLM-R sentencepiece model (AutoTokenizer does not work); xlm-roberta-base's
    # tokenizer.json is built from the byte-identical sentencepiece.bpe.model (same LFS sha) -> same ids, no conversion
    from transformers import XLMRobertaTokenizerFast
    return XLMRobertaTokenizerFast.from_pretrained(TOKENIZER)


def encode(tok, df, bs=20_000):
    ids = []
    for o in range(0, len(df), bs):
        enc = tok(df.text_a.iloc[o:o + bs].tolist(), df.text_b.iloc[o:o + bs].tolist(), truncation="longest_first",
                  max_length=MAX_LEN, return_attention_mask=False, return_token_type_ids=False)
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
def score(model, ids, pad_id, device):
    order = np.argsort([len(s) for s in ids], kind="stable")
    out = np.empty(len(ids), dtype=np.float32)
    for o in range(0, len(ids), EVAL_BS):
        b = order[o:o + EVAL_BS]
        x, m = collate([ids[i] for i in b], pad_id, device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=dev_is_cuda(device)):
            out[b] = model(input_ids=x, attention_mask=m).logits.float().squeeze(-1).cpu().numpy()
    return out


def dev_is_cuda(device):
    return device.type == "cuda"


def read(f, columns=None):
    df = pd.read_parquet(f, columns=columns)
    return df.head(SMOKE).copy() if SMOKE else df


def loco_cfg():
    c = inputs("loco_config.json")
    if not c:
        return None
    cfg = json.load(open(c[0]))
    cfg["groups"] = pd.read_parquet(inputs("loco_groups.parquet")[0]).set_index("pair_id")["grp"]
    return cfg


def worker(k, n_gpu):
    from transformers import BertForSequenceClassification, get_linear_schedule_with_warmup
    dev = torch.device(f"cuda:{k % n_gpu}") if n_gpu else torch.device("cpu")
    if n_gpu:
        torch.cuda.set_device(dev)
    torch.manual_seed(SEED + k)
    rng = np.random.default_rng(SEED + k)
    tok = tokenizer()
    pad = tok.pad_token_id
    model = BertForSequenceClassification.from_pretrained(MODEL, num_labels=1).to(dev)
    files = {n: f for n, f in data_files().items() if n != "loco_groups"}
    cfg = loco_cfg()
    pre = "loco_" if cfg else ""
    prog = {"model": k, "gpu": torch.cuda.get_device_name(dev) if n_gpu else "cpu"}
    prev = [p for p in inputs(f"{pre}ckpt_m{k}.pt") if not p.startswith(W)]
    if prev:                                                 # score-only rerun
        model.load_state_dict({n: t.float() for n, t in torch.load(prev[0], map_location=dev).items()})
        log(k, f"loaded checkpoint {prev[0]} - skip training")
        prog["train"] = "skipped (checkpoint)"
    else:
        tr = read(files["train"])
        if cfg:
            tr = tr[tr.pair_id.map(cfg["groups"]).to_numpy() == cfg["train_grp"][k]].reset_index(drop=True)
        else:
            tr = tr[tr.cf == k].reset_index(drop=True)
        t0 = time.time()
        ids = encode(tok, tr)
        y = tr.label.to_numpy(np.float32)
        log(k, f"train rows {len(tr):,} pos {y.mean():.3f}, tokenized in {time.time() - t0:.0f}s, "
               f"mean len {np.mean([len(s) for s in ids]):.1f}")
        del tr
        steps = math.ceil(len(ids) / BS)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
        sch = get_linear_schedule_with_warmup(opt, int(WARM * steps), steps)
        scaler = torch.amp.GradScaler("cuda", enabled=bool(n_gpu))
        lossf = torch.nn.BCEWithLogitsLoss()
        perm = rng.permutation(len(ids))
        probe = max(50, int(PROBE_FRAC * steps))
        ckpt_every = max(1, steps // N_CKPT)
        n_score = sum(len(read(f, ["cf"])) if cfg else int((read(f, ["cf"]).cf != k).sum())
                      for n, f in files.items() if n != "train")
        model.train()
        t0, run_loss = time.time(), 0.0
        for s in range(steps):
            b = perm[s * BS:(s + 1) * BS]
            x, m = collate([ids[i] for i in b], pad, dev)
            yt = torch.from_numpy(y[b]).to(dev)
            with torch.autocast("cuda", dtype=torch.float16, enabled=bool(n_gpu)):
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
            if s + 1 == min(probe, steps):
                if n_gpu:
                    torch.cuda.synchronize()
                rate = (s + 1) * BS / (time.time() - t0)
                eta_tr = (steps - s - 1) * BS / rate / 60
                eta_sc = n_score / (3 * rate) / 60           # inference ~3x training throughput (measured below)
                prog.update(probe_steps=probe, train_pairs_per_s=round(rate), eta_train_min=round(eta_tr, 1),
                            eta_score_min_est=round(eta_sc, 1), steps=steps)
                log(k, f"PROBE {probe}/{steps} steps ({PROBE_FRAC:.0%}): {rate:.0f} pairs/s -> ETA train "
                       f"{eta_tr:.1f} min + score {n_score:,} pairs ~{eta_sc:.1f} min; loss {run_loss:.4f}")
                json.dump(prog, open(f"{W}/progress_m{k}.json", "w"), indent=2)
            if (s + 1) % 500 == 0:
                log(k, f"step {s + 1}/{steps} loss {run_loss:.4f} {(s + 1) * BS / (time.time() - t0):.0f} pairs/s")
            if (s + 1) % ckpt_every == 0 or s + 1 == steps:
                torch.save({n: t.half() for n, t in model.state_dict().items()}, f"{W}/{pre}ckpt_m{k}.pt")
                prog.update(ckpt_step=s + 1, loss=round(run_loss, 4))
                json.dump(prog, open(f"{W}/progress_m{k}.json", "w"), indent=2)
        prog["train_min"] = round((time.time() - t0) / 60, 1)
        log(k, f"trained in {prog['train_min']} min")
        del ids, y, opt
    model.eval()
    done = {os.path.basename(p)[7 + len(pre):-8] for p in inputs(f"logits_{pre}*.parquet")
            if cfg or not os.path.basename(p).startswith("logits_loco_")}
    for name, f in files.items():
        if name == "train" or name in done:
            continue
        df = read(f)
        if not cfg:
            df = df[df.cf != k].reset_index(drop=True)      # cf == k rows were this model's training S1
        if not len(df):
            continue
        t0 = time.time()
        lg = score(model, encode(tok, df), pad, dev)
        pd.DataFrame({"pair_id": df.pair_id.to_numpy(), f"xenc_l{k}": lg}).to_parquet(f"{W}/part_{pre}{name}_m{k}.parquet")
        rate = len(df) / (time.time() - t0)
        prog.setdefault("score", {})[name] = {"rows": len(df), "pairs_per_s": round(rate)}
        log(k, f"scored {name}: {len(df):,} rows, {rate:.0f} pairs/s")
        json.dump(prog, open(f"{W}/progress_m{k}.json", "w"), indent=2)


def merge():
    from sklearn.metrics import log_loss, roc_auc_score
    metrics = {}
    pre = "loco_" if loco_cfg() else ""
    for name, f in data_files().items():
        parts = [f"{W}/part_{pre}{name}_m{k}.parquet" for k in (0, 1)]
        if name == "train" or not any(os.path.exists(p) for p in parts):
            continue
        import pyarrow.parquet as pq
        have = pq.ParquetFile(f).schema_arrow.names
        base = read(f, [c for c in ("pair_id", "cf", "label") if c in have])
        for k, p in enumerate(parts):
            base = base.merge(pd.read_parquet(p), on="pair_id", how="left") if os.path.exists(p) \
                else base.assign(**{f"xenc_l{k}": np.nan})
        # OOF model for cf in {0,1}; mean of both for cf == -1
        base["xenc_logit"] = np.where(base.cf == 0, base.xenc_l1, np.where(base.cf == 1, base.xenc_l0,
                                                                         base[["xenc_l0", "xenc_l1"]].mean(axis=1)))
        base[["pair_id", "xenc_l0", "xenc_l1", "xenc_logit"]].to_parquet(f"{W}/logits_{pre}{name}.parquet")
        m = {"rows": len(base), "nan": int(base.xenc_logit.isna().sum())}
        if "label" in base.columns:
            for c in ("xenc_l0", "xenc_l1", "xenc_logit"):
                ok = base[c].notna()
                if ok.any() and base.label[ok].nunique() == 2:
                    p = 1 / (1 + np.exp(-base.loc[ok, c].clip(-30, 30)))
                    m[c] = {"auc": round(roc_auc_score(base.label[ok], p), 5),
                            "logloss": round(log_loss(base.label[ok], p), 5)}
        metrics[name] = m
        for p in parts:
            if os.path.exists(p):
                os.remove(p)
        print("[merge]", name, json.dumps(m), flush=True)
    json.dump(metrics, open(f"{W}/{pre}metrics.json", "w"), indent=2)


if __name__ == "__main__":
    import torch.multiprocessing as mp
    t0 = time.time()
    n_gpu = torch.cuda.device_count()
    print(f"GPUs: {[torch.cuda.get_device_name(i) for i in range(n_gpu)]}; inputs: {list(data_files())}", flush=True)
    assert n_gpu >= 1 or SMOKE, "no GPU"
    tokenizer()                                            # warm the HF cache before the workers start
    from transformers import BertForSequenceClassification
    BertForSequenceClassification.from_pretrained(MODEL, num_labels=1)
    if n_gpu >= 2:
        mp.spawn(worker, args=(n_gpu,), nprocs=2, join=True)
    else:
        for k in (0, 1):
            worker(k, n_gpu)
    merge()
    print(f"total {(time.time() - t0) / 60:.1f} min", flush=True)
