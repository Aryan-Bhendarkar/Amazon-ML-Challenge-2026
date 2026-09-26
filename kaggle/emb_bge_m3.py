"""Lane B (monitor note 12): FROZEN zero-shot BAAI/bge-m3 (MIT, ~568M) dense cosines for uncertain-band pairs.
Self-contained (no `ber` import). Kaggle 2x T4, fp16, max_len 64, dense = L2-normalized CLS vector (bge-m3 dense mode).

Inputs (private dataset amlc-emb-data, from pipelines/emb_export.py):
  records.parquet  key, name, addr, full        pairs.parquet  tag, s1_id, cand_id, k1, k2
Per field (name, addr, full): encode every record (sorted by length, DataParallel over the 2 GPUs), keep fp16
vectors in RAM, compute the cosine for every pair, free. Empty texts -> NaN cosine.
Outputs (/kaggle/working): pair_cos.parquet (tag, s1_id, cand_id, emb_cos_name, emb_cos_addr, emb_cos_full),
progress.json (throughput + ETA after a 1% probe), cos_<field>.npy written per field (progressive).
"""
import glob
import json
import os
import time

import numpy as np
import pandas as pd
import torch

MODEL = "BAAI/bge-m3"
MAX_LEN, BS, SEED = 64, 512, 42
W = os.environ.get("EMB_OUT", "/kaggle/working")
IN = os.environ.get("EMB_IN", "/kaggle/input")
SMOKE = int(os.environ.get("EMB_SMOKE", "0"))            # >0: CPU smoke test on the first SMOKE records
FIELDS = ("name", "addr", "full")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def find(name):
    fs = sorted(glob.glob(f"{IN}/**/{name}", recursive=True))
    assert fs, f"{name} not found under {IN}"
    return fs[0]


def main():
    torch.manual_seed(SEED)
    from transformers import AutoModel, AutoTokenizer
    rec = pd.read_parquet(find("records.parquet"))
    pairs = pd.read_parquet(find("pairs.parquet"))
    log(f"inputs: {glob.glob(f'{IN}/**/*.parquet', recursive=True)}; records {len(rec):,} pairs {len(pairs):,}")
    if SMOKE:                                             # smoke: the first SMOKE pairs of every tag + their records
        pairs = pairs.groupby("tag").head(SMOKE)
        rec = rec[rec.key.isin(set(pairs.k1) | set(pairs.k2))]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL, torch_dtype=torch.float16 if dev == "cuda" else torch.float32).to(dev).eval()
    ngpu = torch.cuda.device_count()
    if ngpu > 1:
        model = torch.nn.DataParallel(model)
    log(f"device {dev} x{max(ngpu, 1)}; {torch.cuda.get_device_name(0) if dev == 'cuda' else ''}")
    idx = pd.Series(np.arange(len(rec)), index=rec.key.to_numpy())
    i1, i2 = idx.reindex(pairs.k1).to_numpy(), idx.reindex(pairs.k2).to_numpy()
    assert not (np.isnan(i1.astype(float)).any() or np.isnan(i2.astype(float)).any())
    out = pairs[["tag", "s1_id", "cand_id"]].copy()
    prog = {"records": len(rec), "pairs": len(pairs)}
    t_all = time.time()
    for fi, field in enumerate(FIELDS):
        txt = rec[field].fillna("").to_numpy()
        empty = np.array([len(t) == 0 for t in txt])
        lens = np.array([len(t) for t in txt])
        order = np.argsort(lens, kind="stable")
        emb = np.zeros((len(txt), 1024), dtype=np.float16)
        t0, done = time.time(), 0
        probe_n = max(BS, len(order) // 100)
        with torch.inference_mode():
            for o in range(0, len(order), BS):
                b = order[o:o + BS]
                enc = tok([txt[j] or " " for j in b], max_length=MAX_LEN, truncation=True, padding=True,
                          return_tensors="pt").to(dev)
                h = model(**enc).last_hidden_state[:, 0]
                h = torch.nn.functional.normalize(h.float(), dim=-1)
                emb[b] = h.cpu().numpy().astype(np.float16)
                done += len(b)
                if fi == 0 and done >= probe_n and "rate" not in prog:
                    # length-sorted: the first 1% are the shortest texts, so the ETA is a lower bound (x ~1.5)
                    rate = done / (time.time() - t0)
                    prog.update(rate=round(rate, 1), eta_min_lower=round(3 * len(order) / rate / 60, 1))
                    json.dump(prog, open(f"{W}/progress.json", "w"))
                    log(f"probe: {rate:.0f} texts/s -> ETA >= {prog['eta_min_lower']} min for 3 fields")
                if (o // BS) % 500 == 0:
                    log(f"{field}: {done:,}/{len(order):,} {done / (time.time() - t0):.0f}/s")
        cos = (emb[i1].astype(np.float32) * emb[i2].astype(np.float32)).sum(1)
        cos[empty[i1] | empty[i2]] = np.nan
        np.save(f"{W}/cos_{field}.npy", cos)
        out[f"emb_cos_{field}"] = cos.astype(np.float32)
        prog[f"{field}_sec"] = round(time.time() - t0, 1)
        json.dump(prog, open(f"{W}/progress.json", "w"))
        log(f"{field} done in {time.time() - t0:.0f}s; mean cos {np.nanmean(cos):.3f}")
        del emb
    out.to_parquet(f"{W}/pair_cos.parquet", index=False)
    prog["total_min"] = round((time.time() - t_all) / 60, 1)
    json.dump(prog, open(f"{W}/progress.json", "w"))
    log(f"done {prog}")


if __name__ == "__main__":
    main()
