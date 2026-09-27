"""DM-val (density-matched validation, monitor note 10): re-tune the GLOBAL threshold of a frozen model so that
clean val with band negatives up-weighted to test's band density (ber.dmval) is maximised.

  tune  : python pipelines/dm_val.py --run <features_v1 run> --tag mini
          -> rho / w from the run's test_pred.parquet (label-free) vs mini, DM + clean curves, t_DM
  confirm: python pipelines/dm_val.py --run <run> --tag fold0x --freeze <dm-val mini run>
          -> frozen rho / w / t_DM on fold0x (predictions scored from the cached ctx features), paired bootstraps
  new model: python pipelines/dm_val.py --run <new run> --tag mini --freeze-w <0710 dm-val mini run>   (same rho / w)
Band = [lo, t_ship + 0.2] with t_ship = the run's shipping threshold (decision.json).
"""
import _bootstrap  # noqa: F401

import argparse
import json
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl

import features_v1 as fv
from ber import bag
from ber import ctx_features as cf
from ber import dmval, harness, io, metric, paths
from ber.tracking import Run

GRID = np.round(np.arange(0.60, 0.976, 0.025), 3)


def assign_pl(pred: pl.DataFrame) -> pl.DataFrame:
    return pred.sort("prob", descending=True).unique("cand_id", keep="first")


def score_tag(run: str, cache: str, tag: str, ids: list, threads: int) -> pd.DataFrame:
    art = paths.ART_DIR / run
    model = bag.load_model(art)                  # single model.lgb or seed bag (bag.json)
    feats = json.loads((art / "features.json").read_text())
    fv.CTX_VER = 3
    X = fv.load_cache(cache, tag).filter(pl.col("s1_id").is_in(ids))
    raw = cf.expand_derived(feats)
    X = X.select(list(dict.fromkeys(["s1_id", "cand_id"] + [f for f in raw if f in X.columns])))
    ctx_cols = [f for f in raw if f not in X.columns]
    if ctx_cols:
        X = X.join(pl.read_parquet(fv.ctx_path(cache, tag)).select(["s1_id", "cand_id"] + ctx_cols),
                   on=["s1_id", "cand_id"], how="left")
    X = cf.add_derived(cf.join_emb(X, cache, tag, feats), feats)
    out = []
    for o in range(0, X.height, 4_000_000):
        P = X.slice(o, 4_000_000).select(["s1_id", "cand_id"] + feats).to_pandas()
        out.append(P[["s1_id", "cand_id"]].assign(prob=model.predict(P[feats], num_threads=threads).astype(np.float32)))
    return pd.concat(out, ignore_index=True)


def main(a):
    t0 = time.time()
    os.environ.setdefault("POLARS_MAX_THREADS", str(a.threads))
    art = paths.ART_DIR / a.run
    t_ship = json.loads((art / "decision.json").read_text())["threshold"]
    lo, hi = a.lo, round(t_ship + 0.2, 3)
    ctx = harness.EvalContext.load(a.tag)
    ids = sorted(ctx.ids)
    src = a.freeze or a.freeze_w
    frozen = json.loads((paths.EXP_DIR / src / "metrics.json").read_text())["dm_params"] if src else None
    with Run(f"dm-val-{a.tag}", hypothesis=a.hypothesis or f"DM-val threshold for {a.run} on {a.tag}",
             params=vars(a), tags=["dm-val", "decision"], parent=a.run) as run:
        if a.pred:
            pred = pd.read_parquet(a.pred)
        elif a.tag == "mini" and (art / "val_pred.parquet").exists():
            pred = pd.read_parquet(art / "val_pred.parquet")
        else:
            print(f"[score] {a.tag} with {a.run}", flush=True)
            pred = score_tag(a.run, a.cache, a.tag, ids, a.threads)
            pred.to_parquet(run.art_dir / f"{a.tag}_pred.parquet")
        asg = assign_pl(pl.from_pandas(pred)).to_pandas()
        gt = io.load_gt_pairs(ids)
        own = pd.Series(gt.s1_id.to_numpy(), index=gt.match_id.to_numpy())
        asg["lab"] = own.reindex(asg.cand_id).to_numpy() == asg.s1_id.to_numpy()
        n_true = pd.Series({k: len(ctx.truth.get(k, ())) for k in ids})
        cty = pd.Series(ctx.country)
        # --- rho (label-free test band mass) and w
        if frozen:
            rho, w = frozen["rho"], frozen["w"]
            dm = {**frozen, "frozen_from": src}
        else:
            tp = assign_pl(pl.read_parquet(art / "test_pred.parquet"))
            s1t = pl.read_parquet(cf._norm_file("test", 1, 1), columns=["entity_id", "country"])
            tpb = tp.filter(pl.col("prob").is_between(lo, hi)).join(s1t.rename({"entity_id": "s1_id"}), on="s1_id")
            test_band = tpb.height / s1t.height
            val_band = dmval.band_mass(asg, len(ids), lo, hi)
            inb = asg.prob.between(lo, hi)
            val_pos = float((inb & asg.lab).sum() / len(ids))
            rho = test_band / val_band
            w = dmval.negative_weight(rho, val_band, val_pos)
            byc_test = (tpb.group_by("country").len("n").join(s1t.group_by("country").len("s"), on="country")
                        .with_columns((pl.col("n") / pl.col("s")).alias("m")))
            vb = asg[inb].assign(c=asg.s1_id.map(cty)).groupby("c").size() / cty.reindex(ids).value_counts()
            rho_c = {r["country"]: round(r["m"] / vb.get(r["country"], np.nan), 3) if r["country"] in vb else None
                     for r in byc_test.to_dicts()}
            dm = {"lo": lo, "hi": hi, "t_ship": t_ship, "test_band_per_s1": round(test_band, 5),
                  "val_band_per_s1": round(val_band, 5), "val_band_pos_per_s1": round(val_pos, 5),
                  "val_band_precision": round(val_pos / val_band, 4), "rho": round(rho, 4), "w": round(w, 4),
                  "rho_by_country_REPORT_ONLY": rho_c, "draws": a.draws}
        print(f"[dm] band [{lo},{hi}] rho {rho:.3f} w {w:.3f} {dm}", flush=True)
        # --- curves
        rows = []
        for t in GRID:
            fc = dmval.entity_f05(asg, n_true, t).mean()
            fd = dmval.dm_entity_f05(asg, n_true, t, w, lo, hi, a.draws).mean()
            rows.append((t, fc, fd))
            print(f"  t={t:.3f} clean {fc:.5f} DM {fd:.5f}", flush=True)
        curve = pd.DataFrame(rows, columns=["t", "clean", "dm"])
        curve.to_csv(run.art_dir / "dm_curve.csv", index=False)
        t_dm = frozen["t_dm"] if a.freeze else float(curve.t[curve.dm.idxmax()])
        dm["t_dm"] = t_dm
        res = {"dm_params": dm}
        for name, fn in (("clean", lambda t: dmval.entity_f05(asg, n_true, t)),
                         ("dm", lambda t: dmval.dm_entity_f05(asg, n_true, t, w, lo, hi, a.draws))):
            s_ship, s_dm = fn(t_ship), fn(t_dm)
            A = pd.DataFrame({"s1_id": s_ship.index, "f05": s_ship.to_numpy()})
            B = pd.DataFrame({"s1_id": s_dm.index, "f05": s_dm.to_numpy()})
            res[name] = {"at_t_ship": round(float(s_ship.mean()), 5), "at_t_dm": round(float(s_dm.mean()), 5),
                         "delta_t_dm_vs_ship": metric.paired_bootstrap(A, B),
                         "by_country_at_t_dm": {c: round(float(v), 5) for c, v in s_dm.groupby(cty.reindex(s_dm.index).to_numpy()).mean().items()},
                         "by_country_at_t_ship": {c: round(float(v), 5) for c, v in s_ship.groupby(cty.reindex(s_ship.index).to_numpy()).mean().items()}}
            print(f"[{name}] t_ship {t_ship}: {res[name]['at_t_ship']}  t_dm {t_dm}: {res[name]['at_t_dm']}  "
                  f"delta {res[name]['delta_t_dm_vs_ship']}", flush=True)
        run.log(**res, timing_min=round((time.time() - t0) / 60, 1))
        run.note(f"{a.tag}: rho {rho:.3f}, w {w:.3f}, t_dm {t_dm} (ship {t_ship}); DM {res['dm']['at_t_ship']} -> "
                 f"{res['dm']['at_t_dm']}, clean {res['clean']['at_t_ship']} -> {res['clean']['at_t_dm']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="20260926-0710_aryan-bhendarkar_feat-v1-v1-n2-g1-g2-g3-g4-g5-ctx3")
    ap.add_argument("--cache", default="v1_n2")
    ap.add_argument("--tag", default="mini")
    ap.add_argument("--lo", type=float, default=0.2)
    ap.add_argument("--draws", type=int, default=5)
    ap.add_argument("--freeze", default="", help="dm-val mini run id: reuse its rho / w / t_dm (confirmation)")
    ap.add_argument("--freeze-w", default="", help="dm-val run id: reuse its rho / w, re-tune t (new models)")
    ap.add_argument("--pred", default="", help="reuse a saved predictions parquet (s1_id, cand_id, prob) for this tag")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--hypothesis", default="")
    main(ap.parse_args())
