"""AUDIT C6: stage-2 stack on D's predictions with signals D does not have (XLM-R cross-encoder logits, 2-hop sibling
similarity, 2nd-S1 margin). Trained on MINI labels (validation data; D never trained on it), evaluated FROZEN on
DM-fold0x (disjoint). Threshold: tuned on DM-mini with 2-fold S1-hash cross-fitted stage-2 predictions.
LOCO-style check: stage-2 trained on one country's mini S1 only, scored on the other country's fold0x S1.

    python pipelines/audit_stack.py
"""
import _bootstrap  # noqa: F401

import json

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl

import audit_d as A
from ber.metric import paired_bootstrap

G3 = A.paths.DATA_DIR / "kaggle" / "g3logits"
FEATS = ["lp", "xlmr", "mlm", "hop_full", "hop_name", "hop_n", "margin", "lp2", "name_tset", "addr_tset",
         "house_rel", "cand_addr_empty", "n_cand_s1", "cand_src", "cand_native"]
GRID = np.round(np.arange(0.6, 0.96, 0.025), 3)


def xenc(tag: str) -> pl.DataFrame:
    m = pl.read_parquet(G3 / "g3_up_map.parquet").filter(pl.col("tag") == tag).with_columns(
        pl.col("k1").str.split("|").list.get(1).alias("s1_id"), pl.col("k2").str.split("|").list.get(1).alias("cand_id"))
    out = m.select("pair_id", "s1_id", "cand_id")
    for name, dirs in (("xlmr", ("xlmr_h0", "xlmr_h1")), ("mlm", ("minilm5x",))):
        parts = []
        for d in dirs:
            for f in sorted((G3 / d).glob(f"logits_h*_{tag}.parquet")):
                x = pl.read_parquet(f)
                c = [k for k in x.columns if k.startswith("xenc_h")][0]
                parts.append(x.select("pair_id", pl.col(c).alias("l")))
        L = pl.concat(parts).group_by("pair_id").agg(pl.col("l").mean().alias(name))   # mini/fold0x: mean of halves
        out = out.join(L, on="pair_id", how="left")
    return out.drop("pair_id")


def frame(tag: str) -> pd.DataFrame:
    a = pl.read_parquet(A.OUT / f"{tag}_asg_hop.parquet").join(xenc(tag), on=["s1_id", "cand_id"], how="left")
    eps = 1e-6
    a = a.with_columns((pl.col("prob").clip(eps, 1 - eps) / (1 - pl.col("prob").clip(eps, 1 - eps))).log().alias("lp"),
                       (pl.col("p2").clip(eps, 1 - eps) / (1 - pl.col("p2").clip(eps, 1 - eps))).log().alias("lp2"))
    return a.to_pandas()


def train(df, seed=42):
    params = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=200, feature_fraction=0.9,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=seed, num_threads=6,
                  deterministic=True, force_row_wise=True)
    return lgb.train(params, lgb.Dataset(df[FEATS], df.lab.astype(int), categorical_feature=["house_rel"]), 400)


def sc(asg, s1, p, t, dm=True):
    return A.entity_scores(asg, s1.n_true, p >= t, dm=dm)


def main():
    mini, fx = frame("mini"), frame("fold0x")
    s1m, s1f = A.load("mini")[1], A.load("fold0x")[1]
    print(f"[data] mini {len(mini):,} fold0x {len(fx):,}; xlmr coverage mini {mini.xlmr.notna().mean():.3f}", flush=True)
    # restrict stage-2 to rows with p >= 0.05 (others stay rejected); cross-fit on mini for threshold tuning
    lo = 0.05
    fold = (pd.util.hash_array(mini.s1_id.to_numpy()) % 2).astype(int)
    oof = np.zeros(len(mini))
    for k in (0, 1):
        tr = (fold != k) & (mini.prob >= lo)
        m = train(mini[tr])
        te = fold == k
        oof[te] = m.predict(mini.loc[te, FEATS])
    oof = np.where(mini.prob >= lo, oof, 0.0)
    base_m = sc(mini, s1m, mini.prob.to_numpy(), A.DM["t"]).mean()
    curve = [(t, sc(mini, s1m, oof, t).mean()) for t in GRID]
    t2, best = max(curve, key=lambda x: x[1])
    print(f"[mini] base {base_m:.5f}; stack OOF best t {t2} -> {best:.5f} (Δ {best - base_m:+.5f})", flush=True)
    m = train(mini[mini.prob >= lo])
    pf = np.where(fx.prob >= lo, m.predict(fx[FEATS]), 0.0)
    base = sc(fx, s1f, fx.prob.to_numpy(), A.DM["t"])
    new = sc(fx, s1f, pf, t2)
    base_c, new_c = sc(fx, s1f, fx.prob.to_numpy(), A.DM["t"], dm=False), sc(fx, s1f, pf, t2, dm=False)
    bs = paired_bootstrap(pd.DataFrame({"s1_id": base.index, "f05": base.values}), pd.DataFrame({"s1_id": new.index, "f05": new.values}))
    bc = paired_bootstrap(pd.DataFrame({"s1_id": base_c.index, "f05": base_c.values}), pd.DataFrame({"s1_id": new_c.index, "f05": new_c.values}))
    cty = s1f.country
    byc = {c: round(float(new[cty == c].mean() - base[cty == c].mean()), 5) for c in sorted(cty.unique())}
    imp = dict(sorted(zip(FEATS, m.feature_importance("gain").round(0).tolist()), key=lambda x: -x[1]))
    # LOCO-style: stage-2 trained on one country's mini S1, applied to the other country's fold0x S1
    loco = {}
    cm = mini.country.to_numpy()
    for src in ("US", "India"):
        tgt = "India" if src == "US" else "US"
        ms = train(mini[(mini.prob >= lo) & (cm == src)])
        sel = (fx.country == tgt).to_numpy()
        ids = s1f.index[s1f.country == tgt]
        sub, s1s = fx[sel], s1f.loc[ids]
        ps = np.where(sub.prob >= lo, ms.predict(sub[FEATS]), 0.0)
        b0 = sc(sub, s1s, sub.prob.to_numpy(), A.DM["t"]).mean()
        b1 = sc(sub, s1s, ps, t2).mean()
        loco[f"{src}->{tgt}"] = round(b1 - b0, 5)
    res = {"t_stack": float(t2), "dm_mini_oof_delta": round(best - base_m, 5), "dm_fold0x_delta": round(bs["delta"], 5),
           "dm_ci95": [round(x, 5) for x in bs["ci95"]], "clean_fold0x_delta": round(bc["delta"], 5),
           "clean_ci95": [round(x, 5) for x in bc["ci95"]], "by_country": byc, "loco_delta": loco, "gain": imp}
    res["PASS"] = bool(bs["delta"] >= 0.0005 and bs["ci95"][0] > 0 and min(byc.values()) >= 0 and min(loco.values()) >= -0.001)
    print(json.dumps(res, indent=1), flush=True)
    (A.OUT / "stack_result.json").write_text(json.dumps(res, indent=1))
    m.save_model(str(A.OUT / "stack_model.lgb"))


if __name__ == "__main__":
    main()
