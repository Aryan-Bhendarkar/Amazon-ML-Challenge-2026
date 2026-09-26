"""Lane C label-free France report on a TEST SAMPLE (diagnostic only; nothing is cached or submitted).

Samples --n-s1 test S1 per country, takes their v1_n2 candidate pairs, builds ctx features with statistics
over the FULL test split, scores with a frozen features_v1 model (default: box2 repro of 0710) and applies its
threshold after assign_best_s1 (competition only among the sampled S1: identical for every variant, so
variant deltas are fair; absolute accept rates are slightly optimistic).

France variants:  v1       = shipping (norm_v1 base + ctx, BIZ_WORDS only)
                  v2       = norm v2 base (pipelines/refeat_norm.recompute) + ctx on norm_v2
                  v2biz    = v2 + France-keyed BIZ_WORDS_BY_COUNTRY
US/India: v1 (reference; norm v2 and the biz words do not touch them).

    python pipelines/fr_sample_report.py --n-s1 20000 --threads 3
"""
import _bootstrap  # noqa: F401

import argparse
import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl

import refeat_norm as rn
from ber import ctx_features as cf
from ber import paths
from ber.decision import assign_best_s1
from ber.tracking import Run

CTL = "20260926-1801_aryan-box2_feat-v1-v1-n2-g1-g2-g3-g4-g5-ctx3-ctlrep"


def sample_pairs(n_s1: int) -> tuple[pl.DataFrame, dict]:
    s1 = pl.read_parquet(paths.FEATURE_DIR / "norm_v1_test_s1.parquet", columns=["entity_id", "country"])
    ids, cmap = [], {}
    for c in sorted(s1["country"].unique()):
        x = s1.filter(pl.col("country") == c).sample(n_s1, seed=1)["entity_id"]
        ids.append(x)
        cmap.update(dict.fromkeys(x.to_list(), c))
    ids = pl.concat(ids)
    P = pl.scan_parquet(paths.DATA_DIR / "cands" / "v1_n2" / "test.parquet") \
          .filter(pl.col("s1_id").is_in(ids.implode())).collect()
    return P, cmap


def ctx_feats(P: pl.DataFrame, nv: int, sctx, threads: int, biz: bool) -> pl.DataFrame:
    saved = cf.BIZ_WORDS_BY_COUNTRY
    cf.BIZ_WORDS_BY_COUNTRY = {"france": cf.FR_BIZ_CANDIDATES} if biz else {}
    try:
        F = cf.add_features(cf.attach_norm(P.select("s1_id", "cand_id", "name_tset"), "test", nv), sctx, workers=threads)
    finally:
        cf.BIZ_WORDS_BY_COUNTRY = saved
    new = cf.new_feature_cols(F, ["name_tset"])
    return P.join(F.select(["s1_id", "cand_id"] + new), on=["s1_id", "cand_id"], how="left")


def summarize(X: pd.DataFrame, t: float) -> dict:
    a = assign_best_s1(X[["s1_id", "cand_id", "prob"]])
    acc = a[a.prob >= t]
    n_s1 = X.s1_id.nunique()
    per = acc.groupby("s1_id").size()
    strong = X[(X.name_tset >= 90) & (X.addr_tset >= 90)]
    strong_s1 = set(strong.s1_id)
    acc_s1 = set(acc.s1_id)
    band = a[(a.prob >= 0.3) & (a.prob < t)]
    coloc = X.groupby("s1_id")["s1_addr_self"].first() > 1
    acc_by = {k: round(float(np.mean([s in acc_s1 for s in coloc.index[coloc == v]])), 4)
              for k, v in (("coloc_S1_nonempty", True), ("single_addr_S1_nonempty", False))}
    return {"n_s1": n_s1, "pred_per_s1": round(len(acc) / n_s1, 3), "empty_rate": round(1 - len(acc_s1) / n_s1, 4),
            "share_1match": round(float((per == 1).sum() / n_s1), 4), "share_4plus": round(float((per >= 4).sum() / n_s1), 4),
            "strong_cand_S1": round(len(strong_s1) / n_s1, 4),
            "strong_but_empty": round(len(strong_s1 - acc_s1) / max(len(strong_s1), 1), 4),
            "band_records_per_s1": round(len(band) / n_s1, 3), **acc_by,
            "coloc_S1_share": round(float(coloc.mean()), 4)}


def dists(X: pd.DataFrame) -> dict:
    q = lambda s: [round(float(v), 3) for v in np.nanquantile(s.astype(float), [0.1, 0.5, 0.9])]  # noqa: E731
    return {c: q(X[c]) for c in ["s1_addr_self", "s1_addr_c", "s1_hs_self", "pool_addr_c", "s1_name_self",
                                 "ex_lfrac_cmax", "mi_lfrac_cmax", "ex_ldf_max"]} | \
           {"ex_biz_rate": round(float(X.ex_biz.mean()), 4), "mi_biz_rate": round(float(X.mi_biz.mean()), 4)}


def main(a):
    t0 = time.time()
    art = paths.ART_DIR / a.model
    feats = json.loads((art / "features.json").read_text())
    t = json.loads((art / "decision.json").read_text())["threshold"]
    model = lgb.Booster(model_file=str(art / "model.lgb"))
    P, cmap = sample_pairs(a.n_s1)
    P = P.with_columns(pl.col("s1_id").replace_strict(cmap).alias("_c"))
    print(f"[sample] {P.height:,} pairs {time.time() - t0:.0f}s", flush=True)
    fr = P.filter(pl.col("_c") == "France").drop("_c")
    variants = {}
    sctx1 = cf.SplitContext.build("test", 3, 1)
    variants["v1"] = ctx_feats(P.drop("_c"), 1, sctx1, a.threads, biz=False)
    del sctx1
    print(f"[v1] {time.time() - t0:.0f}s", flush=True)
    fr2 = rn.recompute(fr, fr.schema, "test", 2)
    sctx2 = cf.SplitContext.build("test", 3, 2)
    variants["v2"] = ctx_feats(fr2, 2, sctx2, a.threads, biz=False)
    variants["v2biz"] = ctx_feats(fr2, 2, sctx2, a.threads, biz=True)
    del sctx2
    print(f"[v2, v2biz] {time.time() - t0:.0f}s", flush=True)
    out, prob = {}, {}
    for v, X in variants.items():
        Xp = cf.add_derived(X, feats).to_pandas()
        Xp["prob"] = model.predict(Xp[feats], num_threads=a.threads)
        Xp["country"] = Xp.s1_id.map(cmap)
        for c, g in Xp.groupby("country"):
            if v != "v1" and c != "France":
                continue
            out[f"{c}/{v}"] = {**summarize(g, t), **dists(g)}
        prob[v] = Xp[Xp.country == "France"].set_index(["s1_id", "cand_id"])[["prob", "ex_biz", "mi_biz"]]
    # pair-level shifts vs v1 on France
    sh = {}
    base = prob["v1"]
    for v in ("v2", "v2biz"):
        j = base.join(prob[v], rsuffix="_n", how="inner")
        band = (j.prob >= 0.3) & (j.prob < t)
        sh[v] = {"pairs": len(j), "biz_flag_changed": round(float(((j.ex_biz != j.ex_biz_n) | (j.mi_biz != j.mi_biz_n)).mean()), 4),
                 "band_pairs": int(band.sum()), "band_mean_dprob": round(float((j.prob_n - j.prob)[band].mean()), 4),
                 "cross_t_up": int(((j.prob < t) & (j.prob_n >= t)).sum()), "cross_t_down": int(((j.prob >= t) & (j.prob_n < t)).sum())}
    j = prob["v2"].join(prob["v2biz"], rsuffix="_n", how="inner")
    band = (j.prob >= 0.3) & (j.prob < t)
    sh["v2biz_vs_v2"] = {"biz_flag_changed": round(float(((j.ex_biz != j.ex_biz_n) | (j.mi_biz != j.mi_biz_n)).mean()), 4),
                         "band_mean_dprob": round(float((j.prob_n - j.prob)[band].mean()), 4),
                         "cross_t_up": int(((j.prob < t) & (j.prob_n >= t)).sum()),
                         "cross_t_down": int(((j.prob >= t) & (j.prob_n < t)).sum())}
    flips = j[(j.prob < t) & (j.prob_n >= t)].reset_index()[["s1_id", "cand_id", "prob", "prob_n"]]
    with Run("fr-sample-report", hypothesis=a.hypothesis, params={"n_s1": a.n_s1, "model": a.model, "threshold": t},
             tags=["laneC", "france", "label-free"], parent=a.model) as run:
        run.log(summary=out, fr_shift=sh, timing_min=round((time.time() - t0) / 60, 1))
        raw = pl.concat([pl.read_parquet(paths.PARQUET_DIR / f"test_s{i}.parquet") for i in (1, 2, 3)])
        nm = dict(zip(raw["entity_id"], raw["business_name"] + " | " + raw["business_address"]))
        flips = flips.assign(s1=flips.s1_id.map(nm), cand=flips.cand_id.map(nm))
        flips.to_csv(run.art_dir / "biz_flips_up.tsv", sep="\t", index=False)
        run.note("Biz-word up-flips (v2 -> v2biz), sample of 25:\n\n```\n%s\n```" %
                 flips.sample(min(25, len(flips)), random_state=0)[["prob", "prob_n", "s1", "cand"]].to_string(index=False))
        df = pd.DataFrame(out).T
        print(df.to_string())
        print(json.dumps(sh, indent=1))
        run.note("Label-free test-sample report (20k S1/country, frozen 0710 repro, t=%s).\n\n```\n%s\n```\n\nFR shifts:\n```\n%s\n```"
                 % (t, df.to_string(), json.dumps(sh, indent=1)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-s1", type=int, default=20000)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--model", default=CTL)
    ap.add_argument("--hypothesis", default="FR norm v2 + French BIZ words: label-free effect on FR test sample; FR density shift")
    main(ap.parse_args())
