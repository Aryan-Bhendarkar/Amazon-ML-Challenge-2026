"""Lane G / G3: score the G3 pair set with the G2 cross-encoder checkpoints (Kaggle GPU). Self-contained.

Inputs:
  dataset amlc-g3 (pipelines/xenc_g3_prep.py): pairs.parquet (pair_id, tag, half, r1, r2), records.parquet (rid, text)
  kernel source = the G2 training kernel: ckpt_best_h{k}.pt (fp16 state_dict of the bare model)
Cross-fit: the model of half k scores train pairs with half == 1-k (out-of-fold) and every half == -1 pair
(fold-1 train, mini, fold0x, test). It never scores its own training half.
Outputs (/kaggle/working): logits_h{k}_{tag}.parquet (pair_id, xenc_h{k}) per tag (written as each tag finishes, test
in 1M-pair parts so a time-out keeps partial results), progress_g3.json.
"""
import glob
import json
import os
import threading
import time

import numpy as np
import pandas as pd
import torch

CONFIG = {"model": "microsoft/Multilingual-MiniLM-L12-H384", "halves": [0, 1]}  # @CONFIG@ (generated from xenc_g3.py)
MAX_LEN, EVAL_BS, CHUNK = 96, 1024, 1_000_000
W, IN = "/kaggle/working", "/kaggle/input"
TAGS = ["mini", "train", "fold0x", "test"]
PROG = {}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def find(name):
    fs = [f for f in glob.glob(f"{IN}/**/{name}", recursive=True)]
    assert fs, f"missing input {name}"
    return fs[0]


def tokenizer():
    from transformers import AutoTokenizer, XLMRobertaTokenizerFast
    if "MiniLM" in CONFIG["model"]:
        return XLMRobertaTokenizerFast.from_pretrained("FacebookAI/xlm-roberta-base")
    return AutoTokenizer.from_pretrained(CONFIG["model"])


def load_model(k, dev):
    from transformers import AutoConfig, AutoModelForSequenceClassification, BertForSequenceClassification
    cfg = AutoConfig.from_pretrained(CONFIG["model"], num_labels=1)
    m = BertForSequenceClassification(cfg) if "MiniLM" in CONFIG["model"] else \
        AutoModelForSequenceClassification.from_config(cfg)
    sd = torch.load(find(f"ckpt_best_h{k}.pt"), map_location="cpu")
    missing, unexpected = m.load_state_dict({n: t.float() for n, t in sd.items()}, strict=False)
    assert not unexpected and all("position_ids" in x for x in missing), (missing, unexpected)
    return m.half().to(dev).eval()


@torch.inference_mode()
def score(model, ids, pad, dev):
    order = np.argsort([len(s) for s in ids], kind="stable")
    out = np.empty(len(ids), dtype=np.float32)
    for o in range(0, len(ids), EVAL_BS):
        b = order[o:o + EVAL_BS]
        L = max(len(ids[i]) for i in b)
        x = np.full((len(b), L), pad, dtype=np.int64)
        for j, i in enumerate(b):
            x[j, :len(ids[i])] = ids[i]
        x = torch.from_numpy(x).to(dev, non_blocking=True)
        out[b] = model(input_ids=x, attention_mask=(x != pad).long()).logits.float().squeeze(-1).cpu().numpy()
    return out


def run_half(k, gpu, P, text, tok):
    dev = torch.device(f"cuda:{gpu}")
    torch.cuda.set_device(dev)
    model = load_model(k, dev)
    pad = tok.pad_token_id
    mine = P[(P.half == -1) | (P.half == 1 - k)]
    PROG[k] = {"gpu": torch.cuda.get_device_name(dev), "rows": int(len(mine)), "tags": {}}
    for tag in TAGS:
        d = mine[mine.tag == tag]
        for part, o in enumerate(range(0, len(d), CHUNK)):
            f = f"{W}/logits_h{k}_{tag}_{part:02d}.parquet"
            if os.path.exists(f):
                continue
            c = d.iloc[o:o + CHUNK]
            t0 = time.time()
            enc = tok(text[c.r1.to_numpy()].tolist(), text[c.r2.to_numpy()].tolist(), truncation="longest_first",
                      max_length=MAX_LEN, return_attention_mask=False, return_token_type_ids=False)["input_ids"]
            t1 = time.time()
            lg = score(model, [np.asarray(x, dtype=np.int32) for x in enc], pad, dev)
            pd.DataFrame({"pair_id": c.pair_id.to_numpy(), f"xenc_h{k}": lg}).to_parquet(f)
            rate = len(c) / (time.time() - t1)
            PROG[k]["tags"][f"{tag}_{part:02d}"] = {"rows": len(c), "tok_s": round(t1 - t0), "pairs_per_s": round(rate)}
            log(f"h{k} {tag} part {part}: {len(c):,} pairs, tokenize {t1 - t0:.0f}s, score {rate:.0f} pairs/s")
            json.dump(PROG, open(f"{W}/progress_g3.json", "w"), indent=2)


if __name__ == "__main__":
    t0 = time.time()
    n_gpu = torch.cuda.device_count()
    assert n_gpu >= 1, "no GPU"
    print(f"CONFIG {json.dumps(CONFIG)}; GPUs {[torch.cuda.get_device_name(i) for i in range(n_gpu)]}", flush=True)
    P = pd.read_parquet(find("pairs.parquet"))
    R = pd.read_parquet(find("records.parquet"))
    text = np.empty(int(R.rid.max()) + 1, dtype=object)
    text[R.rid.to_numpy()] = R.text.to_numpy()
    del R
    tok = tokenizer()
    halves = CONFIG["halves"]
    log(f"{len(P):,} pairs; halves {halves}")
    if n_gpu >= 2 and len(halves) == 2:            # one model per GPU (threads: GPU work releases the GIL)
        th = [threading.Thread(target=run_half, args=(h, i, P, text, tok)) for i, h in enumerate(halves)]
        [t.start() for t in th]
        [t.join() for t in th]
    else:
        for h in halves:
            run_half(h, 0, P, text, tok)
    # merge parts per half and tag
    for h in halves:
        for tag in TAGS:
            parts = sorted(glob.glob(f"{W}/logits_h{h}_{tag}_*.parquet"))
            if parts:
                pd.concat([pd.read_parquet(p) for p in parts]).to_parquet(f"{W}/logits_h{h}_{tag}.parquet")
                [os.remove(p) for p in parts]
    PROG["total_min"] = round((time.time() - t0) / 60, 1)
    json.dump(PROG, open(f"{W}/progress_g3.json", "w"), indent=2)
    log(f"done in {PROG['total_min']} min")
