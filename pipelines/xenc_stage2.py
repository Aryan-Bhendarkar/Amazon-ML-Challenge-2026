"""Stage 5 gate: does the cross-encoder logit (`xenc_logit`) improve the stage-2 decision layer?

Control and treatment share EVERYTHING (stage-1 frames, K-model, isotonic, seeds); the only difference is the
extra stage-2 column `xenc_logit`, masked to NaN outside the stage-1 uncertain band BAND_LO < prob < BAND_HI
(identical rule on train OOF / mini / fold0x / test, so missingness means the same thing everywhere).

xenc logits: artifacts/kaggle/<slug>/logits_<name>.parquet (pair_id -> xenc_l0, xenc_l1, xenc_logit), joined back
through data/kaggle/xenc_map/<name>.parquet (pair_id -> s1_id, cand_id). Train tag = score_train (cross-fitted OOF;
es rows = mean of both models); eval tags = mean of both models.

Stage-1 frames come from a decision_v1 run (--decision-run: oof_train.parquet, val_pred_stage1.parquet, stage1.lgb
for the confirm tag) or are computed with decision_v1.stage1 (--stage1-run etc., slow).

    python pipelines/xenc_stage2.py --decision-run <decision_v1 run on v1_n1> --confirm-tag fold0x --loco --threads 4
"""
import _bootstrap  # noqa: F401

import argparse
import gc
import json
import os
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
from sklearn.isotonic import IsotonicRegression

import decision_v1 as dv
import features_v1 as fv
from ber import harness, io, metric, paths
from ber import setdecision as sd
from ber.decision import assign_best_s1
from ber.tracking import Run

MAP = paths.DATA_DIR / "kaggle" / "xenc_map"
BAND_LO, BAND_HI = 0.02, 0.98
TAG_FILES = {"train": ["s1band_train"], "mini": ["band_mini", "s1band_mini"], "fold0x": ["s1band_fold0x"],
             "test": ["s1band_test"]}


def xenc_table(xdirs, names: list[str], col: str, prefix: str = "") -> pd.DataFrame:
    """xdirs: list of pulled kernel-output dirs, searched in order for logits_<prefix><name>.parquet."""
    parts = []
    for n in names:
        found = [Path(d) / f"logits_{prefix}{n}.parquet" for d in xdirs if (Path(d) / f"logits_{prefix}{n}.parquet").exists()]
        if not found:
            raise FileNotFoundError(f"logits_{prefix}{n}.parquet not in {xdirs} (kernel not run on {n} yet)")
        lp = found[0]
        mp = pl.read_parquet(MAP / f"{n}.parquet", columns=["pair_id", "s1_id", "cand_id"])
        parts.append(mp.join(pl.read_parquet(lp, columns=["pair_id", col]), on="pair_id")
                     .select("s1_id", "cand_id", pl.col(col).cast(pl.Float32).alias("xenc_logit")))
    return pl.concat(parts).unique(["s1_id", "cand_id"], keep="first").to_pandas()


def attach(df: pd.DataFrame, X: pd.DataFrame, tag: str, log: dict) -> pd.DataFrame:
    out = df.merge(X, on=["s1_id", "cand_id"], how="left", validate="one_to_one")
    band = ((out.prob > BAND_LO) & (out.prob < BAND_HI)).to_numpy()
    out["xenc_logit"] = out["xenc_logit"].where(band)
    cov = float(out.loc[band, "xenc_logit"].notna().mean()) if band.any() else float("nan")
    log[tag] = {"rows": len(out), "band_rows": int(band.sum()), "band_coverage": round(cov, 5)}
    print(f"[xenc:{tag}] band rows {band.sum():,}, coverage {cov:.4f}", flush=True)
    return out


def confirm_frame(art: Path, cache: str, tag: str, feats: list[str], threads: int) -> pd.DataFrame:
    """Stage-1 probs on the confirm tag with the decision run's stage1.lgb (decision_v1 does not save them).
    Cached to artifacts/<decision_run>/stage1_<tag>.parquet (shared with xenc_export s1band)."""
    cache_p = art / f"stage1_{tag}.parquet"
    if cache_p.exists():
        return pd.read_parquet(cache_p)
    model = lgb.Booster(model_file=str(art / "stage1.lgb"))
    df = fv.load_cache(cache, tag)
    new = [c for c in feats if c not in df.columns]
    if new:
        f = pl.read_parquet(fv.ctx_path(cache, tag), columns=["s1_id", "cand_id"] + new)
        df = df.join(f, on=["s1_id", "cand_id"], how="left")
    keep = [c for c in dv.S2_PAIR if c in feats]
    out = []
    for o in range(0, df.height, 4_000_000):
        c = fv.to_pandas(df.slice(o, 4_000_000), ["s1_id", "cand_id"] + feats)
        out.append(c[["s1_id", "cand_id"] + keep].assign(
            prob=model.predict(c[feats], num_threads=threads).astype(np.float32)))
    out = pd.concat(out, ignore_index=True)
    out.to_parquet(cache_p)
    return out


def fit_set_level(tr: pd.DataFrame, tr_truth: dict, es_ids: set, pmin: float, threads: int, km=None):
    """decision_v1.main's set level (K-model + stage-2 + isotonic). Pass km to reuse the K-model (xenc-free)."""
    R_tr = tr[tr.prob >= pmin]
    P_tr = sd.s1_profile(R_tr, tr_truth.keys())
    is_es = P_tr.index.isin(list(es_ids))
    kfeat = list(P_tr.columns)
    if km is None:
        y = sd.oracle_k(R_tr, tr_truth).loc[P_tr.index, "k_star"].clip(upper=dv.C_K - 1).to_numpy()
        km = dv.train_k_model(P_tr[~is_es], y[~is_es], P_tr[is_es], y[is_es], threads)
    S_tr = dv.s2_frame(R_tr, P_tr)
    s2cols = [c for c in S_tr.columns if c not in ("s1_id", "cand_id", "label", "is_es", "rank")]
    lab = np.fromiter((c in tr_truth.get(s, ()) for s, c in zip(S_tr.s1_id, S_tr.cand_id)), bool, len(S_tr))
    es2 = S_tr.s1_id.isin(es_ids).to_numpy()
    s2m = dv.train_s2(S_tr.loc[~es2, s2cols], lab[~es2], S_tr.loc[es2, s2cols], lab[es2], threads)
    A_tr = assign_best_s1(R_tr)
    lab_a = np.fromiter((c in tr_truth.get(s, ()) for s, c in zip(A_tr.s1_id, A_tr.cand_id)), bool, len(A_tr))
    iso = IsotonicRegression(out_of_bounds="clip").fit(A_tr.prob.to_numpy(), lab_a)
    gain = dict(sorted(zip(s2cols, s2m.feature_importance("gain").round(1).tolist()), key=lambda x: -x[1])[:12])
    return km, s2m, iso, kfeat, s2cols, {"s2_best_iter": s2m.best_iteration, "s2_gain_top": gain}


def boot(ent_a: dict, ent_b: dict, truth: dict) -> dict:
    a = metric.per_entity_scores(ent_a, truth)
    b = metric.per_entity_scores(ent_b, truth)
    return metric.paired_bootstrap(a, b)


def run_arm(name, tr, ev, cf_, ctx, cctx, tr_truth, es_ids, a, km=None):
    t0 = time.time()
    km, s2m, iso, kfeat, s2cols, info = fit_set_level(tr, tr_truth, es_ids, a.pmin, a.threads, km)
    print(f"[{name}] set level fitted {time.time() - t0:.0f}s; s2 top gain {list(info['s2_gain_top'])[:6]}")
    rules, ent, params, _ = dv.evaluate(ev[ev.prob >= a.pmin], ctx, km, s2m, iso, kfeat, s2cols, a.threads)
    chosen = max((k for k in rules if isinstance(rules[k], dict)), key=lambda k: rules[k]["f05"])
    res = {"info": info, "rules": rules, "chosen": chosen, "params": params, "models": (km, s2m, kfeat, s2cols)}
    res["ent"] = ent
    if cf_ is not None:
        crules, cent, _, _ = dv.evaluate(cf_[cf_.prob >= a.pmin], cctx, km, s2m, iso, kfeat, s2cols, a.threads,
                                         params=params)
        res["confirm_rules"], res["confirm_ent"] = crules, cent
    return res, km


LOCO_COL = {"US": "xenc_l0", "India": "xenc_l1"}     # amlc-xenc-loco-run: model 0 trained on US only, 1 on India


def loco(tr, ev, ctx, tr_truth, es_ids, cmap, a, ev_src_only=None):
    """Stage-2 trained on one country's train S1, rules tuned+scored on the other country's eval S1.
    Arms: control (no xenc) | xenc_indomain (target logits from the xenc that saw both countries: optimistic) |
    xenc_srconly (target logits from an xenc trained on the SOURCE country only = the France situation: stage 2
    learnt to trust an in-domain xenc, then meets an xenc that never saw the country)."""
    out = {}
    ctry_tr, ctry_ev = tr.s1_id.map(cmap), ev.s1_id.map(cmap)
    for src in sorted(set(ctry_ev.dropna())):
        for tgt in sorted(set(ctry_ev.dropna()) - {src}):
            sub_ids = {s for s in ctx.ids if ctx.country.get(s) == tgt}
            sctx = harness.EvalContext(ctx.subset, sub_ids, {s: ctx.truth[s] for s in sub_ids},
                                       {s: tgt for s in sub_ids})
            trs = tr[ctry_tr == src]
            tt = {s: v for s, v in tr_truth.items() if cmap.get(s) == src}
            r = {}
            km = None
            arms = [("control", True, None), ("xenc_indomain", False, None)]
            if ev_src_only is not None:
                arms.append(("xenc_srconly", False, LOCO_COL[src]))
            for arm, drop, scol in arms:
                X = trs.drop(columns="xenc_logit") if drop else trs
                E = ev[ctry_ev == tgt]
                E = E.drop(columns="xenc_logit") if drop else E
                if scol:
                    E = E.drop(columns="xenc_logit").merge(ev_src_only[["s1_id", "cand_id", scol]]
                                                           .rename(columns={scol: "xenc_logit"}),
                                                           on=["s1_id", "cand_id"], how="left")
                    E["xenc_logit"] = E["xenc_logit"].where((E.prob > BAND_LO) & (E.prob < BAND_HI))
                res, km = run_arm(f"loco {src}->{tgt} {arm}", X, E, None, sctx, None, tt, es_ids, a, km)
                r[arm] = {"f05": res["rules"][res["chosen"]]["f05"], "rule": res["chosen"],
                          "s2_kmodel": res["rules"]["s2_kmodel"]["f05"], "s2_thr": res["rules"]["s2_thr"]["f05"]}
            for arm in r.copy():
                if arm != "control":
                    r[f"delta_{arm}"] = round(r[arm]["f05"] - r["control"]["f05"], 5)
            out[f"{src}->{tgt}"] = r
            print(f"[loco] {src}->{tgt}: {r}", flush=True)
    return out


def main(a):
    t0 = time.time()
    os.environ.setdefault("POLARS_MAX_THREADS", str(a.threads))
    xdir = a.xenc_dir.split(",")
    ctx = harness.EvalContext.load(a.subset)
    cctx = harness.EvalContext.load(a.confirm_tag) if a.confirm_tag else None
    art = paths.ART_DIR / a.decision_run
    feats = json.loads((art / "features.json").read_text())
    tr = pd.read_parquet(art / "oof_train.parquet")
    ev = pd.read_parquet(art / "val_pred_stage1.parquet")
    ev = ev[ev.s1_id.isin(ctx.ids)]
    cf_ = None
    if a.confirm_tag:
        cf_ = confirm_frame(art, a.cache, a.confirm_tag, feats, a.threads)
        cf_ = cf_[cf_.s1_id.isin(cctx.ids)]
    cov = {}
    tr = attach(tr, xenc_table(xdir, TAG_FILES["train"], a.train_col), "train", cov)
    ev = attach(ev, xenc_table(xdir, TAG_FILES[a.eval_tag], a.eval_col), a.eval_tag, cov)
    if cf_ is not None:
        cf_ = attach(cf_, xenc_table(xdir, TAG_FILES[a.confirm_tag], a.eval_col), a.confirm_tag, cov)
    tr_truth = io.load_gt_sets(set(pl.read_parquet(fv.CANDS / a.cache / "train.parquet", columns=["s1_id"])
                                   ["s1_id"].unique().to_list()))
    es_ids = set(tr.loc[tr.is_es, "s1_id"])
    gc.collect()
    with Run(f"xenc-stage2-{a.cache}", hypothesis=a.hypothesis or
             "cross-encoder logit on the uncertain band as a stage-2 feature beats the same stage 2 without it",
             params=vars(a), tags=["model", "xenc", "decision"], parent=a.decision_run) as run:
        run.log(xenc_coverage=cov)
        ctrl, km = run_arm("control", tr.drop(columns="xenc_logit"), ev.drop(columns="xenc_logit"),
                           None if cf_ is None else cf_.drop(columns="xenc_logit"), ctx, cctx, tr_truth, es_ids, a)
        trt, _ = run_arm("xenc", tr, ev, cf_, ctx, cctx, tr_truth, es_ids, a, km)
        comp = {"mini_vs_shipping_thr": boot(ctrl["ent"]["thr"], trt["ent"][trt["chosen"]], ctx.truth),
                "mini_chosen": boot(ctrl["ent"][ctrl["chosen"]], trt["ent"][trt["chosen"]], ctx.truth),
                "mini_s2_kmodel": boot(ctrl["ent"]["s2_kmodel"], trt["ent"]["s2_kmodel"], ctx.truth)}
        if cf_ is not None:
            comp["confirm_vs_shipping_thr"] = boot(ctrl["confirm_ent"]["thr"], trt["confirm_ent"][trt["chosen"]],
                                                   cctx.truth)
            comp["confirm_chosen"] = boot(ctrl["confirm_ent"][trt["chosen"]], trt["confirm_ent"][trt["chosen"]],
                                          cctx.truth)
        rep = metric.report(trt["ent"][trt["chosen"]], ctx.truth, ctx.country)
        rep["rule"] = trt["chosen"]
        run.log(val=rep, subset=ctx.subset, control={"rules": ctrl["rules"], "chosen": ctrl["chosen"]},
                xenc={"rules": trt["rules"], "chosen": trt["chosen"], "info": trt["info"]},
                control_info=ctrl["info"], bootstrap=comp, decision_params=trt["params"])
        if cf_ is not None:
            run.log(confirm={"tag": a.confirm_tag, "control": ctrl["confirm_rules"], "xenc": trt["confirm_rules"]})
        metric.per_entity_scores(trt["ent"][trt["chosen"]], ctx.truth).to_parquet(run.art_dir / "val_entity_scores.parquet")
        metric.per_entity_scores(ctrl["ent"][ctrl["chosen"]], ctx.truth).to_parquet(
            run.art_dir / "val_entity_scores_control.parquet")
        km_, s2m, kfeat, s2cols = trt["models"]
        s2m.save_model(str(run.art_dir / "stage2.lgb"))
        km_.save_model(str(run.art_dir / "k_model.lgb"))
        (run.art_dir / "decision.json").write_text(json.dumps(
            {"rule": trt["chosen"], "params": trt["params"], "kfeat": kfeat, "s2cols": s2cols, "pmin": a.pmin,
             "xenc_band": [BAND_LO, BAND_HI], "xenc_eval_col": a.eval_col}))
        if a.loco:
            try:
                folds = io.load_folds()
                cmap = dict(zip(folds.s1_id, folds.country))
                del folds
                so = None
                if a.loco_xenc_dir:
                    so = pd.concat([xenc_table(a.loco_xenc_dir.split(","), TAG_FILES[a.eval_tag], c, "loco_")
                                    .rename(columns={"xenc_logit": c}).set_index(["s1_id", "cand_id"])
                                    for c in ("xenc_l0", "xenc_l1")], axis=1).reset_index()
                run.log(loco=loco(tr, ev, ctx, tr_truth, es_ids, cmap, a, so))
            except Exception as e:  # noqa: BLE001
                run.log(loco_error=repr(e))
                print("[loco] FAILED", repr(e))
        run.log(timing_min=round((time.time() - t0) / 60, 1))
        d = comp["mini_vs_shipping_thr"]
        note = (f"{ctx.subset}: shipping thr={ctrl['rules']['thr']['f05']:.5f}; control best {ctrl['chosen']}="
                f"{ctrl['rules'][ctrl['chosen']]['f05']:.5f} -> xenc "
                f"{trt['chosen']}={trt['rules'][trt['chosen']]['f05']:.5f}; bootstrap {json.dumps(d)}")
        if cf_ is not None:
            note += (f" | CONFIRM {a.confirm_tag} ({trt['chosen']}): control "
                     f"{ctrl['confirm_rules'][trt['chosen']]['f05']:.5f} -> xenc "
                     f"{trt['confirm_rules'][trt['chosen']]['f05']:.5f} (shipping thr "
                     f"{ctrl['confirm_rules']['thr']['f05']:.5f}); {json.dumps(comp['confirm_vs_shipping_thr'])}")
        run.note(note)
        print(note)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--decision-run", required=True, help="decision_v1 run on the same cache (stage-1 frames)")
    ap.add_argument("--xenc-dir", default=",".join(str(paths.ART_DIR / "kaggle" / d) for d in
                                                    ("amlc-xenc-score", "amlc-xenc")), help="comma list, searched in order")
    ap.add_argument("--loco-xenc-dir", default=",".join(str(paths.ART_DIR / "kaggle" / d) for d in
                                                         ("amlc-xenc-loco-score", "amlc-xenc-loco-run")))
    ap.add_argument("--cache", default="v1_n1")
    ap.add_argument("--eval-tag", default="mini")
    ap.add_argument("--subset", default="mini", choices=["micro", "mini", "fold0"])
    ap.add_argument("--confirm-tag", default="")
    ap.add_argument("--train-col", default="xenc_logit", help="train tag: OOF (cross-fitted) logit")
    ap.add_argument("--eval-col", default="xenc_logit", help="eval tags: xenc_logit (mean) | xenc_l0 | xenc_l1")
    ap.add_argument("--pmin", type=float, default=0.01)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--loco", action="store_true")
    ap.add_argument("--hypothesis", default="")
    main(ap.parse_args())
