"""Train + evaluate the pair model on a cached candidate/feature set (built by pipelines/build_cache.py
or blocking_v1). No blocking here -> model/decision experiments take minutes.

Cache: data/cands/<cand_ver>/train.parquet (role 'train' = folds 2-4 sample, 'es' = fold-1 early stopping)
       data/cands/<cand_ver>/<subset>.parquet (role 'eval'); columns s1_id, cand_id, label, role, features.
Protocol: LightGBM params identical to baseline_v0; threshold tuned on the eval subset (harness);
paired bootstrap vs --parent on identical entities; --loco adds leave-one-country-out
(train on country A only, eval on B at B's tuned threshold AND at A's threshold -> threshold drift).

    python pipelines/train_eval.py --cand-ver keys_v0 --subset mini --name repro-keys-v0 \
        --parent 20260925-1236_aryan_baseline-v0-keys-lgbm --loco
"""
import _bootstrap  # noqa: F401

import argparse
import json
import resource
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from ber import harness, io, metric, paths
from ber.decision import assign_best_s1, threshold_matches, tune_threshold
from ber.tracking import Run

SEED = 42
BASE_RUN = "20260925-1236_aryan_baseline-v0-keys-lgbm"
CATEGORICAL = ["house_rel", "legal_rel", "state_rel", "cand_kind"]
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=100,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              verbose=-1, seed=SEED, num_threads=0, deterministic=True, force_row_wise=True)


def rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def load_cache(cand_ver: str, name: str) -> pd.DataFrame:
    return pd.read_parquet(paths.DATA_DIR / "cands" / cand_ver / f"{name}.parquet")


def fit(tr: pd.DataFrame, feats: list[str], max_rounds: int, log_every: int = 200) -> lgb.Booster:
    is_es = (tr.role == "es").to_numpy()
    cats = [c for c in CATEGORICAL if c in feats]
    dtr = lgb.Dataset(tr.loc[~is_es, feats], tr.loc[~is_es, "label"].astype(int), categorical_feature=cats)
    des = lgb.Dataset(tr.loc[is_es, feats], tr.loc[is_es, "label"].astype(int), reference=dtr)
    return lgb.train(PARAMS, dtr, max_rounds, valid_sets=[des],
                     callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(log_every)])


def predict(model: lgb.Booster, ev: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    return ev[["s1_id", "cand_id"]].assign(prob=model.predict(ev[feats], num_threads=0).astype(np.float32))


def f05_at(pred: pd.DataFrame, truth: dict, t: float) -> float:
    return metric.macro_f05(threshold_matches(assign_best_s1(pred), t), truth)


def loco(tr, ev, feats, ctx, max_rounds) -> dict:
    """Train on one country, evaluate on each other one. Countries are discovered from the data."""
    c_tr = tr.s1_id.map(ctx_country_all(tr.s1_id))
    c_ev = ev.s1_id.map(ctx.country)
    out = {}
    for a in sorted(c_tr.dropna().unique()):
        m = fit(tr[(c_tr == a).to_numpy()], feats, max_rounds, log_every=0)
        pr = predict(m, ev, feats)
        truth_a = {s: t for s, t in ctx.truth.items() if ctx.country.get(s) == a}
        t_a, f_a, _ = tune_threshold(assign_best_s1(pr[(c_ev == a).to_numpy()]), truth_a)
        for b in sorted(c_ev.dropna().unique()):
            if b == a:
                continue
            truth_b = {s: t for s, t in ctx.truth.items() if ctx.country.get(s) == b}
            pb = pr[(c_ev == b).to_numpy()]
            t_b, f_b, _ = tune_threshold(assign_best_s1(pb), truth_b)
            out[f"{a}->{b}"] = {"f05_tuned": f_b, "t_tuned": t_b, "f05_at_src_t": f05_at(pb, truth_b, t_a),
                                "t_src": t_a, "src_in_country_f05": f_a, "best_iter": m.best_iteration}
            print(f"[loco] {a}->{b}: F0.5 {f_b:.4f} @t={t_b} | at src t={t_a}: {out[f'{a}->{b}']['f05_at_src_t']:.4f}",
                  flush=True)
    return out


_COUNTRY_ALL = None


def ctx_country_all(ids: pd.Series) -> dict:
    global _COUNTRY_ALL
    if _COUNTRY_ALL is None:
        f = io.load_folds()
        f = f[f.s1_id.isin(set(ids))]
        _COUNTRY_ALL = dict(zip(f.s1_id, f.country))
    return _COUNTRY_ALL


def main(a):
    t0 = time.time()
    feats = json.loads((paths.ART_DIR / BASE_RUN / "features.json").read_text()) if not a.feats else a.feats.split(",")
    ctx = harness.EvalContext.load(a.subset)
    tr = load_cache(a.cand_ver, "train")
    ev = load_cache(a.cand_ver, a.subset)
    missing = [c for c in feats if c not in tr.columns]
    assert not missing, f"cache lacks features {missing}"
    assert set(ev.s1_id) <= ctx.ids, "eval cache has S1 outside the subset"
    if not a.smoke:     # smoke caches may reuse eval S1 for training (plumbing only, never a result)
        assert set(tr.s1_id).isdisjoint(ctx.ids), "train/eval S1 overlap"
    with Run(a.name, hypothesis=a.hypothesis, parent=a.parent, tags=a.tags.split(","),
             params={"cand_ver": a.cand_ver, "subset": a.subset, "n_feats": len(feats), "lgb": PARAMS,
                     "max_rounds": a.max_rounds, "loco": a.loco}) as run:
        print(f"[data] train {len(tr):,} pairs ({tr.s1_id.nunique():,} S1), eval {len(ev):,} pairs; "
              f"load {time.time() - t0:.0f}s rss {rss_gb():.1f} GB", flush=True)
        model = fit(tr, feats, a.max_rounds)
        model.save_model(str(run.art_dir / "model.lgb"))
        (run.art_dir / "features.json").write_text(json.dumps(feats))
        imp = dict(sorted(zip(feats, model.feature_importance("gain").round(1).tolist()), key=lambda x: -x[1]))
        is_es = (tr.role == "es").to_numpy()
        run.log(train_pairs=int((~is_es).sum()), train_pos_rate=float(tr.label[~is_es].mean()),
                best_iter=model.best_iteration, feature_gain=imp)
        pred = predict(model, ev, feats)
        # blocking numbers of this candidate set (n_pool unknown here -> no reduction ratio)
        harness.log_blocking(run, ev[["s1_id", "cand_id"]], ctx)
        out = harness.log_predictions(run, pred, ctx)
        (run.art_dir / "decision.json").write_text(json.dumps({"threshold": out["val"]["threshold"]}))
        if a.parent:
            pa_ = paths.ART_DIR / a.parent / "val_entity_scores.parquet"
            if pa_.exists():
                bs = metric.paired_bootstrap(pd.read_parquet(pa_),
                                             pd.read_parquet(run.art_dir / "val_entity_scores.parquet"))
                run.log(vs_parent=bs)
                print(f"[bootstrap vs {a.parent}] {bs}", flush=True)
        if a.loco:
            run.log(loco=loco(tr, ev, feats, ctx, a.max_rounds))
        run.log(timing_min=round((time.time() - t0) / 60, 1), peak_rss_gb=round(rss_gb(), 1))
        v = out["val"]
        run.note(f"{a.subset} F0.5={v['f05_macro']:.4f} by_country={ {k: round(x, 4) for k, x in v['f05_by_country'].items()} } "
                 f"P={v['pair_precision']:.4f} R={v['pair_recall']:.4f} t={v['threshold']}. Top feats: {list(imp)[:8]}")
        print(f"done in {(time.time() - t0) / 60:.1f} min, peak rss {rss_gb():.1f} GB -> {run.run_id}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand-ver", required=True)
    ap.add_argument("--subset", default="mini", choices=["micro", "mini", "fold0"])
    ap.add_argument("--name", required=True)
    ap.add_argument("--hypothesis", default="")
    ap.add_argument("--parent", default=None)
    ap.add_argument("--tags", default="model")
    ap.add_argument("--feats", default=None, help="comma list; default = baseline features.json")
    ap.add_argument("--max-rounds", type=int, default=3000)
    ap.add_argument("--loco", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="plumbing test: skip the train/eval disjointness check")
    main(ap.parse_args())
