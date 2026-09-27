"""AUDIT part A: bucket D's remaining loss (1 - F0.5) on fold0x (clean and DM) at t = 0.825, with examples.

    python pipelines/audit_errors.py --tag fold0x
Attribution per S1: singleton with FP -> 'singleton_FP' (loss 1); non-singleton with tp = 0 -> 'nonsingleton_zero'
(split by selection empty / only-FP and by the fate of its true records); partial S1: fp_loss = F(no FP) - F,
fn_loss = 1 - F(no FP), fn_loss split over missing true records by fate (blocking miss / lost to another S1 /
in-candidate reject p < t). DM: band negatives weighted by w (expected copies) inside the same formula.
"""
import _bootstrap  # noqa: F401

import argparse
import json

import numpy as np
import pandas as pd
import polars as pl

import audit_d as A
from ber import paths


def f05(tp, fp, nt):
    f = np.where(tp > 0, 1.25 * tp / np.maximum(1.25 * tp + 0.25 * (nt - tp) + fp, 1e-12), 0.0)
    return np.where(nt == 0, (fp == 0).astype(float), f)


def main(a):
    asg, s1 = A.load(a.tag)
    miss = pd.read_parquet(A.OUT / f"{a.tag}_miss.parquet")
    t, lo, hi, w = A.DM["t"], A.DM["lo"], A.DM["hi"], A.DM["w"]
    sel = asg.prob.to_numpy() >= t
    lab = asg.lab.to_numpy()
    nt_s = s1.n_true
    # sanity: reproduce dm_val numbers
    clean = A.entity_scores(asg, nt_s, sel, dm=False).mean()
    dm = A.entity_scores(asg, nt_s, sel, dm=True).mean()
    print(f"[repro] clean {clean:.5f} DM {dm:.5f} (dm_val fold0x: clean 0.98548, DM 0.98497)", flush=True)
    band_neg = (~lab) & (asg.prob.to_numpy() >= lo) & (asg.prob.to_numpy() <= hi)
    out = {"tag": a.tag, "t": t, "clean": round(clean, 5), "dm": round(dm, 5)}
    # FP row subtypes
    fp_rows = asg[sel & ~lab].copy()
    other_owner = fp_rows.owner_any.notna()
    near_word = (fp_rows.extra_tok.fillna(0) + fp_rows.missing_tok.fillna(0) > 0) & (fp_rows.addr_tset >= 90)
    house_diff = fp_rows.house_rel.isin([1, 2, 4, 5, 6])
    fp_rows["fp_type"] = np.select(
        [other_owner & (fp_rows.name_tset >= 90) & (fp_rows.addr_tset >= 90),
         other_owner,
         ~other_owner & house_diff & (fp_rows.name_tset >= 90),
         ~other_owner & near_word,
         ~other_owner],
        ["owned_by_other_S1_same_name_addr (chain/co-located twin)", "owned_by_other_S1_other",
         "unowned_house_diff (near-copy house±k)", "unowned_extra_or_missing_word_same_addr (near-copy word)",
         "unowned_other"], "x")
    fp_rows["w"] = np.where(band_neg[sel & ~lab], w, 1.0)
    fates = miss[miss.fate != "in_cand_assigned"].copy()
    rej = miss[(miss.fate == "in_cand_assigned")].merge(asg[["cand_id", "prob"]], on="cand_id", how="left")
    rej = rej[rej.prob < t].assign(fate="in_cand_reject_p<t")
    fn_rows = pd.concat([fates[["s1_id", "cand_id", "fate"]], rej[["s1_id", "cand_id", "fate"]]])
    for mode in ("clean", "dm"):
        cw = np.where(band_neg, w, 1.0) if mode == "dm" else np.ones(len(asg))
        g = pd.DataFrame({"s1_id": asg.s1_id, "tp": (sel & lab).astype(float), "fp": (sel & ~lab) * cw}) \
              .groupby("s1_id")[["tp", "fp"]].sum().reindex(s1.index, fill_value=0.0)
        tp, fp, nt = g.tp.to_numpy(), g.fp.to_numpy(), s1.n_true.to_numpy().astype(float)
        f = f05(tp, fp, nt)
        f_nofp = f05(tp, np.zeros_like(fp), nt)
        loss = 1 - f
        total = loss.sum()
        cat = np.select([(nt == 0) & (fp > 0), (nt > 0) & (tp == 0) & (fp == 0), (nt > 0) & (tp == 0) & (fp > 0)],
                        ["singleton_FP", "nonsingleton_empty", "nonsingleton_only_FP"], "partial")
        fp_loss = np.where(cat == "partial", f_nofp - f, np.where(cat == "singleton_FP", 1.0,
                           np.where(cat == "nonsingleton_only_FP", 0.0, 0.0)))
        fn_loss = loss - fp_loss                    # nonsingleton_empty / only_FP -> all FN-side (recall 0)
        per = pd.DataFrame({"s1_id": s1.index, "cat": cat, "loss": loss, "fp_loss": fp_loss, "fn_loss": fn_loss,
                            "country": s1.country.to_numpy()})
        res = {"total_loss": round(total, 2), "mean_loss": round(total / len(s1), 5)}
        res["by_cat_%"] = (per.groupby("cat").loss.sum() / total * 100).round(2).to_dict()
        res["fp_side_%"] = round(per.fp_loss.sum() / total * 100, 2)
        res["fn_side_%"] = round(per.fn_loss.sum() / total * 100, 2)
        res["by_country_mean_loss"] = per.groupby("country").loss.mean().round(5).to_dict()
        # FP side by subtype: distribute each S1's fp_loss over its FP rows by weight
        fr = fp_rows.merge(per[["s1_id", "fp_loss"]], on="s1_id")
        wsum = fr.groupby("s1_id").w.transform("sum") if mode == "dm" else fr.groupby("s1_id").w.transform("size")
        share = (fr.w if mode == "dm" else 1.0) / wsum
        fr["l"] = fr.fp_loss * share
        res["fp_by_type_%"] = (fr.groupby("fp_type").l.sum() / total * 100).round(2).to_dict()
        res["fp_by_source_%"] = (fr.groupby(np.where(fr.cand_src == 1, "S3", "S2")).l.sum() / total * 100).round(2).to_dict()
        res["fp_cand_addr_empty_%"] = round(fr.loc[fr.cand_addr_empty == 1, "l"].sum() / total * 100, 2)
        # FN side by fate
        fn = fn_rows.merge(per[["s1_id", "fn_loss"]], on="s1_id")
        fn["l"] = fn.fn_loss / fn.groupby("s1_id").cand_id.transform("size")
        res["fn_by_fate_%"] = (fn.groupby("fate").l.sum() / total * 100).round(2).to_dict()
        src = np.where(fn.cand_id.str.startswith("S3"), "S3", "S2")
        res["fn_by_source_%"] = (fn.groupby(src).l.sum() / total * 100).round(2).to_dict()
        res["singleton_FP_by_type_%"] = (fr[fr.s1_id.isin(per.s1_id[per.cat == "singleton_FP"])]
                                         .groupby("fp_type").l.sum() / total * 100).round(2).to_dict()
        out[mode] = res
        if mode == "dm":
            per.to_parquet(A.OUT / f"{a.tag}_perS1_loss.parquet")
    # examples (raw text) - 5 per FP type, 5 per FN fate
    raw = pl.concat([pl.read_parquet(paths.PARQUET_DIR / f"train_s{i}.parquet") for i in (1, 2, 3)]).to_pandas()
    txt = dict(zip(raw.entity_id, raw.business_name.fillna("") + " | " + raw.business_address.fillna("")))
    ex = {}
    for ty, g in fp_rows.groupby("fp_type"):
        s = g.sample(min(5, len(g)), random_state=0)
        ex[f"FP::{ty}"] = [f"p={r.prob:.3f} m={r.margin:.3f} S1[{txt.get(r.s1_id, '')}] REC[{txt.get(r.cand_id, '')}] "
                           f"owner={'other S1' if pd.notna(r.owner_any) else 'none'}" for r in s.itertuples()]
    pmap = dict(zip(asg.cand_id, asg.prob))
    smap = dict(zip(asg.cand_id, asg.s1_id))
    for ty, g in fn_rows.groupby("fate"):
        s = g.sample(min(5, len(g)), random_state=0)
        ex[f"FN::{ty}"] = [f"S1[{txt.get(r.s1_id, '')}] REC[{txt.get(r.cand_id, '')}] p_asg={pmap.get(r.cand_id, float('nan')):.3f}"
                           f"{' ASSIGNED TO [' + txt.get(smap.get(r.cand_id), '') + ']' if smap.get(r.cand_id) not in (None, r.s1_id) else ''}"
                           for r in s.itertuples()]
    out["examples"] = ex
    (A.OUT / f"{a.tag}_errors.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in out.items() if k != "examples"}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="fold0x")
    main(ap.parse_args())
