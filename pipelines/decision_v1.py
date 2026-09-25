"""EXP-C1 decision v1 (strategy_v2 P2): OOF stage-1 -> set-level cardinality + stage-2 relative model.

1. Stage 1 = the features_v1 recipe (base + chosen ctx groups) on the candidate cache.
   OOF probs over the train tag (4 folds by S1 hash); eval-tag probs from a model on the whole train tag.
2. assign_best_s1 within each tag (same partial-competition protocol for train and eval).
3. Set level, trained on train-tag OOF, early stopping on the role=='es' S1s:
     K-model   multiclass over k* (oracle best prefix length, classes 0..C-1) -> choose_k (expected F0.5)
     S2-model  stage-2 pair classifier on relative features + S1 profile + stage-1 prob
4. Rules compared on the eval tag (all tuned on mini -> confirm on fold0 minus mini):
     thr        global threshold on stage-1 (reference)
     iso_ef     expected-F0.5 top-k on isotonic-calibrated stage-1 probs
     kmodel     top-k with k from the K-model (+ prob floor tuned)
     s2_thr     global threshold on stage-2 probs
     s2_kmodel  K-model k applied to the stage-2 ranking (+ floor)

    python pipelines/decision_v1.py --cache keys_v0 --groups G1,G2,G3,G4,G5 --threads 6
"""
import _bootstrap  # noqa: F401

import argparse
import gc
import json
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
from sklearn.isotonic import IsotonicRegression

import features_v1 as fv
from ber import ctx_features as cf
from ber import harness, io, metric, paths
from ber import setdecision as sd
from ber.decision import assign_best_s1, expected_f05_matches, threshold_matches, tune_threshold
from ber.tracking import Run

SEED = 42
N_OOF = 4
C_K = 8                          # k* classes 0..6 and "7+"
S2_PAIR = ["name_tset", "addr_tset", "cand_addr_empty", "house_rel", "name_rank_in_s1", "addr_rank_in_s1",
           "n_cand_s1", "cand_src"]


def stage1(a, log=print):
    """Returns (oof_train_pred, eval_pred, train_frame_meta, feats) with prob columns."""
    groups = [] if a.groups == "none" else a.groups.split(",")
    base = fv.base_feats()
    tr_pl = fv.load_cache(a.cache, "train")
    ev_pl = fv.load_cache(a.cache, a.eval_tag)
    cf_pl = fv.load_cache(a.cache, a.confirm_tag) if a.confirm_tag else None
    base += [c for c in a.extra_base.split(",") if c and c in tr_pl.columns]
    new = []
    if groups:
        sctx = None
        if not (fv.ctx_path(a.cache, "train").exists() and fv.ctx_path(a.cache, a.eval_tag).exists()):
            sctx = cf.SplitContext.build("train")
        f_tr = fv.featurize_tag(a.cache, "train", tr_pl, sctx, a.chunks, a.threads)
        f_ev = fv.featurize_tag(a.cache, a.eval_tag, ev_pl, sctx, max(1, a.chunks // 4), a.threads)
        if cf_pl is not None:
            f_cf = fv.featurize_tag(a.cache, a.confirm_tag, cf_pl, sctx, a.chunks, a.threads)
        new = [c for c in f_tr.columns if c not in ("s1_id", "cand_id") and fv._group_of(c) in groups]
        tr_pl = tr_pl.join(f_tr.select(["s1_id", "cand_id"] + new), on=["s1_id", "cand_id"], how="left")
        ev_pl = ev_pl.join(f_ev.select(["s1_id", "cand_id"] + new), on=["s1_id", "cand_id"], how="left")
        if cf_pl is not None:
            cf_pl = cf_pl.join(f_cf.select(["s1_id", "cand_id"] + new), on=["s1_id", "cand_id"], how="left")
            del f_cf
        del f_tr, f_ev, sctx
    feats = base + new
    cats = fv.CAT_BASE + [c for c in fv.CAT_NEW if c in new]
    keep_pair = [c for c in S2_PAIR if c in feats]
    tr_pl = tr_pl.with_columns((pl.col("s1_id").hash(seed=5) % N_OOF).cast(pl.Int8).alias("oof"))
    tr = fv.to_pandas(tr_pl, ["s1_id", "cand_id", "label", "is_es", "oof"] + feats)
    ev = fv.to_pandas(ev_pl, ["s1_id", "cand_id"] + feats)
    del tr_pl, ev_pl
    gc.collect()
    oof = np.zeros(len(tr), dtype=np.float32)
    iters = []
    for k in range(N_OOF):
        t0 = time.time()
        m_in = (tr["oof"] != k).to_numpy()
        model = fv.train_lgb(tr[m_in], feats, cats, a.threads, lr=a.lr)
        oof[~m_in] = model.predict(tr.loc[~m_in, feats], num_threads=a.threads)
        iters.append(model.best_iteration)
        log(f"  oof fold {k}: best_iter {model.best_iteration}, {time.time() - t0:.0f}s")
        del model
        gc.collect()
    if a.stage1_run:                     # reuse the features_v1 model trained on the whole train tag
        art = paths.ART_DIR / a.stage1_run
        assert json.loads((art / "features.json").read_text()) == feats, "stage-1 run has different features"
        model = lgb.Booster(model_file=str(art / "model.lgb"))
    else:
        model = fv.train_lgb(tr, feats, cats, a.threads, lr=a.lr)
    ev_prob = model.predict(ev[feats], num_threads=a.threads).astype(np.float32)
    tr_out = tr[["s1_id", "cand_id", "label", "is_es"] + keep_pair].assign(prob=oof)
    ev_out = ev[["s1_id", "cand_id"] + keep_pair].assign(prob=ev_prob)
    cf_out = None
    if cf_pl is not None:
        cf_out = []                      # score in slices (fold0x is ~25M pairs)
        for o in range(0, cf_pl.height, 4_000_000):
            c = fv.to_pandas(cf_pl.slice(o, 4_000_000), ["s1_id", "cand_id"] + feats)
            cf_out.append(c[["s1_id", "cand_id"] + keep_pair].assign(
                prob=model.predict(c[feats], num_threads=a.threads).astype(np.float32)))
        cf_out = pd.concat(cf_out, ignore_index=True)
        del cf_pl
    return tr_out, ev_out, cf_out, feats, {"oof_iters": iters, "full_iter": model.best_iteration,
                                   "stage1_run": a.stage1_run}, model


def train_k_model(Xtr, ytr, Xes, yes, threads):
    params = dict(objective="multiclass", num_class=C_K, learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
                  feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1,
                  seed=SEED, num_threads=threads, deterministic=True)
    d = lgb.Dataset(Xtr, ytr)
    return lgb.train(params, d, 2000, valid_sets=[lgb.Dataset(Xes, yes, reference=d)],
                     callbacks=[lgb.early_stopping(50, verbose=False)])


def train_s2(Xtr, ytr, Xes, yes, threads):
    params = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
                  feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1,
                  seed=SEED, num_threads=threads, deterministic=True)
    d = lgb.Dataset(Xtr, ytr)
    return lgb.train(params, d, 3000, valid_sets=[lgb.Dataset(Xes, yes, reference=d)],
                     callbacks=[lgb.early_stopping(100, verbose=False)])


PCOLS = ["p1", "p2", "p3", "gap1", "gap2", "n_ge50", "n_ge90", "p_sum", "s2_ge50", "s3_ge50", "n_assigned"]


def s2_frame(R: pd.DataFrame, prof: pd.DataFrame) -> pd.DataFrame:
    """Stage-2 rows from PRE-assignment candidates (S1-local: identical meaning on val and test)."""
    return sd.rel_features(R).join(prof[PCOLS].add_prefix("s_"), on="s1_id")


def owned(R: pd.DataFrame, matches: dict, p_col="prob") -> dict:
    """Drop selected records that another S1 wants more (each record has at most one owner)."""
    A = assign_best_s1(R[["s1_id", "cand_id", p_col]].rename(columns={p_col: "prob"}))
    win = set(zip(A.s1_id, A.cand_id))
    out = {s: [c for c in cs if (s, c) in win] for s, cs in matches.items()}
    return {s: cs for s, cs in out.items() if cs}


def level2(R: pd.DataFrame, ids, km, s2m, kfeat, s2cols, threads):
    """S1-local set features -> k_hat; stage-2 probs. R = pre-assignment stage-1 candidates (prob >= pmin)."""
    P = sd.s1_profile(R, ids)
    k_hat = pd.Series(sd.choose_k(km.predict(P[kfeat], num_threads=threads)), index=P.index)
    S = s2_frame(R, P)
    S["prob2"] = s2m.predict(S[s2cols], num_threads=threads).astype(np.float32)
    return P, k_hat, S


FLOORS = (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5)


def evaluate(R, ctx, km, s2m, iso, kfeat, s2cols, threads, params=None):
    """All rules on one eval set. params=None -> tune on this set (mini); else apply frozen params (fold0x)."""
    tune = params is None
    params = dict(params or {})
    P, k_hat, S = level2(R, ctx.ids, km, s2m, kfeat, s2cols, threads)
    A = assign_best_s1(R)
    A2 = assign_best_s1(S[["s1_id", "cand_id", "prob2"]].rename(columns={"prob2": "prob"}))
    R2 = S[["s1_id", "cand_id", "prob2"]]
    f = lambda m: metric.macro_f05(m, ctx.truth)  # noqa: E731
    cands = {
        "thr": lambda t: threshold_matches(A, t),
        "s2_thr": lambda t: threshold_matches(A2, t),
        "kmodel": lambda fl: owned(R, sd.topk_matches(R, k_hat, floor=fl)),
        "s2_kmodel": lambda fl: owned(R2, sd.topk_matches(R2, k_hat, p_col="prob2", floor=fl), p_col="prob2"),
    }
    if tune:
        grid = np.round(np.arange(0.3, 0.96, 0.025), 3)
        params["thr"] = float(max(grid, key=lambda t: f(cands["thr"](t))))
        params["s2_thr"] = float(max(grid, key=lambda t: f(cands["s2_thr"](t))))
        params["kmodel"] = float(max(FLOORS, key=lambda fl: f(cands["kmodel"](fl))))
        params["s2_kmodel"] = float(max(FLOORS, key=lambda fl: f(cands["s2_kmodel"](fl))))
    rules, ent = {}, {}
    for name, fn in cands.items():
        ent[name] = fn(params[name])
    A_iso = A.assign(prob=iso.predict(A.prob.to_numpy()))
    ent["iso_ef"] = expected_f05_matches(A_iso[A_iso.prob >= 0.02], max_k=12)
    for name, m in ent.items():
        rep = metric.report(m, ctx.truth, ctx.country)
        rules[name] = {"f05": round(rep["f05_macro"], 5), "by_country": rep["f05_by_country"],
                       "P": round(rep["pair_precision"], 4), "R": round(rep["pair_recall"], 4),
                       "singleton_acc": rep["singleton_acc"], "param": params.get(name)}
        print(f"  [{ctx.subset}] {name:10s} {rep['f05_macro']:.5f} {rep['f05_by_country']} param={params.get(name)}")
    O = sd.oracle_k(R, ctx.truth)
    rules["oracle_prefix"] = round(float(O.f_star.mean()), 5)
    rules["k_acc"] = round(float((k_hat.loc[O.index].to_numpy() == O.k_star.clip(upper=C_K - 1).to_numpy()).mean()), 4)
    return rules, ent, params, S


def main(a):
    t0 = time.time()
    os.environ.setdefault("POLARS_MAX_THREADS", str(a.threads))
    ctx = harness.EvalContext.load(a.subset)
    cctx = harness.EvalContext.load(a.confirm_tag) if a.confirm_tag else None
    with Run(f"decision-v1-{a.cache}", hypothesis=a.hypothesis or
             "S1-local set-level decision (k* multiclass + stage-2 relative) beats a global threshold",
             params=vars(a), tags=["decision", "model"], parent=a.parent) as run:
        print("[stage 1 + OOF]")
        tr, ev, cf_, feats, info, s1_model = stage1(a)
        s1_model.save_model(str(run.art_dir / "stage1.lgb"))
        (run.art_dir / "features.json").write_text(json.dumps(feats))
        ev = ev[ev.s1_id.isin(ctx.ids)]
        run.log(stage1=info, n_feats=len(feats), train_pairs=len(tr), eval_pairs=len(ev))
        tr.to_parquet(run.art_dir / "oof_train.parquet")
        ev.to_parquet(run.art_dir / "val_pred_stage1.parquet")
        # truth for ALL train-tag S1 (including S1 without candidates in the cache)
        tr_truth = io.load_gt_sets(set(fv.load_cache(a.cache, "train")["s1_id"].unique().to_list()))
        es_ids = set(tr.loc[tr.is_es, "s1_id"])
        print("[set level]")
        R_tr = tr[tr.prob >= a.pmin]
        P_tr = sd.s1_profile(R_tr, tr_truth.keys())
        y = sd.oracle_k(R_tr, tr_truth).loc[P_tr.index, "k_star"].clip(upper=C_K - 1).to_numpy()
        is_es = P_tr.index.isin(list(es_ids))
        kfeat = list(P_tr.columns)
        km = train_k_model(P_tr[~is_es], y[~is_es], P_tr[is_es], y[is_es], a.threads)
        S_tr = s2_frame(R_tr, P_tr)
        s2cols = [c for c in S_tr.columns if c not in ("s1_id", "cand_id", "label", "is_es", "rank")]
        lab_tr = np.fromiter((c in tr_truth.get(s, ()) for s, c in zip(S_tr.s1_id, S_tr.cand_id)), bool, len(S_tr))
        es2 = S_tr.s1_id.isin(es_ids).to_numpy()
        s2m = train_s2(S_tr.loc[~es2, s2cols], lab_tr[~es2], S_tr.loc[es2, s2cols], lab_tr[es2], a.threads)
        run.log(k_model={"best_iter": km.best_iteration},
                s2={"best_iter": s2m.best_iteration, "feats": s2cols,
                    "gain": dict(sorted(zip(s2cols, s2m.feature_importance("gain").round(1).tolist()),
                                        key=lambda x: -x[1])[:15])})
        A_tr = assign_best_s1(R_tr)
        lab_a = np.fromiter((c in tr_truth.get(s, ()) for s, c in zip(A_tr.s1_id, A_tr.cand_id)), bool, len(A_tr))
        iso = IsotonicRegression(out_of_bounds="clip").fit(A_tr.prob.to_numpy(), lab_a)
        del S_tr, P_tr, A_tr
        gc.collect()

        print("[rules: tune on eval tag]")
        R_ev = ev[ev.prob >= a.pmin]
        rules, ent, params, S_ev = evaluate(R_ev, ctx, km, s2m, iso, kfeat, s2cols, a.threads)
        chosen = max((k for k in rules if isinstance(rules[k], dict)), key=lambda k: rules[k]["f05"])
        rep = metric.report(ent[chosen], ctx.truth, ctx.country)
        rep["rule"] = chosen
        run.log(rules=rules, val=rep, subset=ctx.subset, decision_params=params)
        for name, m in ent.items():
            metric.per_entity_scores(m, ctx.truth).to_parquet(run.art_dir / f"val_entity_scores_{name}.parquet")
        metric.per_entity_scores(ent[chosen], ctx.truth).to_parquet(run.art_dir / "val_entity_scores.parquet")
        S_ev.to_parquet(run.art_dir / "val_stage2.parquet")
        km.save_model(str(run.art_dir / "k_model.lgb"))
        s2m.save_model(str(run.art_dir / "stage2.lgb"))
        (run.art_dir / "decision.json").write_text(json.dumps(
            {"rule": chosen, "params": params, "kfeat": kfeat, "s2cols": s2cols, "pmin": a.pmin}))
        note = (f"{ctx.subset}: " + ", ".join(f"{k}={v['f05']:.4f}" for k, v in rules.items() if isinstance(v, dict))
                + f"; prefix oracle {rules['oracle_prefix']:.4f}. Chosen on {ctx.subset}: {chosen}.")
        if cf_ is not None:
            print(f"[rules: frozen params on {a.confirm_tag}]")
            cf_ = cf_[cf_.s1_id.isin(cctx.ids)]
            crules, cent, _, _ = evaluate(cf_[cf_.prob >= a.pmin], cctx, km, s2m, iso, kfeat, s2cols, a.threads,
                                          params=params)
            run.log(confirm={"tag": a.confirm_tag, "rules": crules})
            for name, m in cent.items():
                metric.per_entity_scores(m, cctx.truth).to_parquet(
                    run.art_dir / f"confirm_entity_scores_{name}.parquet")
            note += (f" CONFIRM {a.confirm_tag}: " + ", ".join(f"{k}={v['f05']:.4f}" for k, v in crules.items()
                                                                if isinstance(v, dict)))
        run.log(timing_min=round((time.time() - t0) / 60, 1))
        run.note(note)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="keys_v0")
    ap.add_argument("--eval-tag", default="mini")
    ap.add_argument("--subset", default="mini", choices=["micro", "mini", "fold0"])
    ap.add_argument("--confirm-tag", default="", help="e.g. fold0x: apply the mini-tuned rules frozen")
    ap.add_argument("--groups", default="G1,G2,G3,G4,G5")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--chunks", type=int, default=8)
    ap.add_argument("--extra-base", default=",".join(fv.EXTRA_BASE))
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--pmin", type=float, default=0.01, help="drop stage-1 probs below this")
    ap.add_argument("--stage1-run", default=None, help="features_v1 run whose model scores the eval tags")
    ap.add_argument("--parent", default=None)
    ap.add_argument("--hypothesis", default="")
    main(ap.parse_args())
