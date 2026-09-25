"""Stage 5 interim: gate model (v1_n1) + cross-encoder logit stacked on the gate's uncertain band.

In the band GATE_LO < p_gate < GATE_HI the prob is replaced by sigmoid(a*logit(p_gate) + b*xenc_logit + c); the
3 stack weights are fitted on fold-1 (es) pairs ONLY (valid.parquet; xenc never trained on fold 1), so mini is
untouched until scoring. Threshold tuned on mini exactly like every other run (harness.log_predictions).
GATE_LO = 0.05 because v1_test saves test gate probs >= 0.05, so the same rule applies to test unchanged.

    python pipelines/xenc_gate_stack.py [--xenc-dir artifacts/kaggle/amlc-xenc] [--loco-dir artifacts/kaggle/amlc-xenc-loco-run]
    python pipelines/xenc_gate_stack.py --apply-test <this run_id> --xenc-dir artifacts/kaggle/amlc-xenc-score
        -> artifacts/<run_id>/test_matches.parquet (+ gate test_candidates.parquet = the scored set) for make_submission
"""
import _bootstrap  # noqa: F401

import argparse
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
import pyarrow.parquet as pq
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from ber import harness, io, metric, paths
from ber.decision import assign_best_s1, threshold_matches, tune_threshold
from ber.tracking import Run

GATE_RUN = "20260925-2015_aryan-bhendarkar_gate-blocking-v1-n1"
MAP = paths.DATA_DIR / "kaggle" / "xenc_map"
CACHE = paths.DATA_DIR / "cands" / "v1_n1"
GATE_LO, GATE_HI = 0.05, 0.98


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def xenc(xdir: Path, name: str, col: str) -> pd.DataFrame:
    mp = pl.read_parquet(MAP / f"{name}.parquet", columns=["pair_id", "s1_id", "cand_id"])
    return mp.join(pl.read_parquet(xdir / f"logits_{name}.parquet"), on="pair_id") \
             .select("s1_id", "cand_id", pl.col(col).alias("xenc")).to_pandas()


def valid_frame(xdir: Path, col: str, threads: int) -> pd.DataFrame:
    """fold-1 (es) valid pairs with exact gate probs (the gate used fold 1 only for early stopping) + labels."""
    art = paths.ART_DIR / GATE_RUN
    model = lgb.Booster(model_file=str(art / "model.lgb"))
    feats = json.loads((art / "features.json").read_text())
    V = xenc(xdir, "valid", col)
    keys = pl.from_pandas(V[["s1_id", "cand_id"]])
    pf = pq.ParquetFile(CACHE / "train.parquet")
    parts = []
    for rg in range(pf.num_row_groups):
        df = pl.from_arrow(pf.read_row_group(rg, columns=["s1_id", "cand_id", "label", "role"] + feats))
        df = df.filter(pl.col("role") == "es").join(keys, on=["s1_id", "cand_id"], how="semi")
        if df.height:
            parts.append(df.select("s1_id", "cand_id", "label").to_pandas().assign(
                prob=model.predict(df.select(feats).to_pandas(), num_threads=threads)))
    g = pd.concat(parts, ignore_index=True)
    return g.merge(V, on=["s1_id", "cand_id"], how="inner", validate="one_to_one")


def in_band(p):
    return (p > GATE_LO) & (p < GATE_HI)


def fit_stack(V: pd.DataFrame) -> LogisticRegression:
    B = V[in_band(V.prob)]
    return LogisticRegression(C=10.0).fit(np.c_[logit(B.prob), B.xenc], B.label.astype(int))


def apply_stack(pred: pd.DataFrame, X: pd.DataFrame, lr: LogisticRegression) -> tuple[pd.DataFrame, dict]:
    d = pred.merge(X, on=["s1_id", "cand_id"], how="left", validate="one_to_one")
    band = in_band(d.prob).to_numpy()
    ok = band & d.xenc.notna().to_numpy()
    newp = d.prob.to_numpy().astype(np.float64).copy()
    newp[ok] = lr.predict_proba(np.c_[logit(d.prob.to_numpy()[ok]), d.xenc.to_numpy()[ok]])[:, 1]
    out = d[["s1_id", "cand_id"]].assign(prob=newp.astype(np.float32))
    return out, {"band_rows": int(band.sum()), "band_coverage": round(float(ok.sum() / max(band.sum(), 1)), 5)}


def f05(pred: pd.DataFrame, truth: dict) -> tuple[float, float, dict]:
    a = assign_best_s1(pred)
    t, f, _ = tune_threshold(a, truth)
    return f, t, threshold_matches(a, t)


def apply_test(a):
    art = paths.ART_DIR / a.apply_test
    d = json.loads((art / "decision.json").read_text())
    lr = LogisticRegression()
    lr.coef_, lr.intercept_, lr.classes_ = np.array(d["coef"]), np.array(d["intercept"]), np.array([0, 1])
    pred = pd.read_parquet(paths.ART_DIR / GATE_RUN / "test_pred.parquet")
    new, cov = apply_stack(pred, xenc(Path(a.xenc_dir), "gband_test", d["xenc_col"]), lr)
    print("[test] coverage", cov)
    assert cov["band_coverage"] > 0.999, cov
    m = threshold_matches(assign_best_s1(new), d["threshold"])
    pd.DataFrame([(s, x) for s, xs in m.items() for x in xs], columns=["s1_id", "match_id"]) \
      .to_parquet(art / "test_matches.parquet")
    new.to_parquet(art / "test_pred.parquet")
    print(f"[test] {sum(map(len, m.values())):,} matches for {len(m):,} S1 -> {art / 'test_matches.parquet'}; "
          f"candidates = {paths.ART_DIR / GATE_RUN / 'test_candidates.parquet'}")


def main(a):
    if a.apply_test:
        return apply_test(a)
    t0 = time.time()
    xdir = Path(a.xenc_dir)
    ctx = harness.EvalContext.load("mini")
    gate = pd.read_parquet(paths.ART_DIR / GATE_RUN / "val_pred.parquet")
    V = valid_frame(xdir, a.col, a.threads)
    lr = fit_stack(V)
    Vb = V[in_band(V.prob)]
    auc_v = {"gate": roc_auc_score(Vb.label, Vb.prob), "xenc": roc_auc_score(Vb.label, Vb.xenc),
             "stack": roc_auc_score(Vb.label, lr.predict_proba(np.c_[logit(Vb.prob), Vb.xenc])[:, 1])}
    print("[valid band]", len(Vb), {k: round(v, 4) for k, v in auc_v.items()}, "coef", lr.coef_, lr.intercept_)
    X = xenc(xdir, "band_mini", a.col)
    pred, cov = apply_stack(gate, X, lr)
    with Run("xenc-gate-stack", hypothesis=a.hypothesis or
             "cross-encoder logit stacked on the gate's uncertain band (0.05<p<0.98) beats the gate alone",
             params={**vars(a), "gate_band": [GATE_LO, GATE_HI], "stack_fit": "fold-1 es valid pairs"},
             tags=["model", "xenc"], parent=GATE_RUN) as run:
        run.log(stack={"coef": lr.coef_.round(4).tolist(), "intercept": lr.intercept_.round(4).tolist(),
                       "valid_band_rows": len(Vb), "valid_band_auc": {k: round(v, 5) for k, v in auc_v.items()}},
                xenc_coverage=cov)
        out = harness.log_predictions(run, pred, ctx)
        base = pd.read_parquet(paths.ART_DIR / GATE_RUN / "val_entity_scores.parquet")
        new = pd.read_parquet(run.art_dir / "val_entity_scores.parquet")
        run.log(bootstrap_vs_gate=metric.paired_bootstrap(base, new))
        (run.art_dir / "decision.json").write_text(json.dumps(
            {"threshold": out["val"]["threshold"], "gate_band": [GATE_LO, GATE_HI], "xenc_col": a.col,
             "coef": lr.coef_.tolist(), "intercept": lr.intercept_.tolist()}))
        # LOCO (stack layer): fit on one country's es pairs, score the other country's mini S1
        folds = io.load_folds()
        cmap = dict(zip(folds.s1_id, folds.country))
        del folds
        loco = {}
        for src in ("US", "India"):
            tgt = "India" if src == "US" else "US"
            lr_s = fit_stack(V[V.s1_id.map(cmap) == src])
            ids = {s for s in ctx.ids if ctx.country.get(s) == tgt}
            tt = {s: ctx.truth[s] for s in ids}
            g = gate[gate.s1_id.isin(ids)]
            fg, tg, _ = f05(g, tt)
            fs, ts, _ = f05(apply_stack(g, X, lr_s)[0], tt)
            loco[f"stack {src}->{tgt}"] = {"gate": round(fg, 5), "stack": round(fs, 5), "delta": round(fs - fg, 5)}
        if a.loco_dir:   # xenc trained on ONE country (xenc_l0 = US-trained, xenc_l1 = India-trained)
            ld = Path(a.loco_dir)
            for col, src, tgt in (("xenc_l0", "US", "India"), ("xenc_l1", "India", "US")):
                mp = pl.read_parquet(MAP / "valid.parquet", columns=["pair_id", "s1_id", "cand_id"])
                Vl = V.drop(columns="xenc").merge(
                    mp.join(pl.read_parquet(ld / "logits_loco_valid.parquet"), on="pair_id")
                    .select("s1_id", "cand_id", pl.col(col).alias("xenc")).to_pandas(), on=["s1_id", "cand_id"])
                lr_s = fit_stack(Vl[Vl.s1_id.map(cmap) == src])
                Xl = pl.read_parquet(MAP / "band_mini.parquet", columns=["pair_id", "s1_id", "cand_id"]) \
                       .join(pl.read_parquet(ld / "logits_loco_band_mini.parquet"), on="pair_id") \
                       .select("s1_id", "cand_id", pl.col(col).alias("xenc")).to_pandas()
                ids = {s for s in ctx.ids if ctx.country.get(s) == tgt}
                tt = {s: ctx.truth[s] for s in ids}
                g = gate[gate.s1_id.isin(ids)]
                fg, _, _ = f05(g, tt)
                fs, _, _ = f05(apply_stack(g, Xl, lr_s)[0], tt)
                Xb = Xl.merge(g, on=["s1_id", "cand_id"])
                Xb = Xb[in_band(Xb.prob)]
                lab = np.fromiter((c in tt.get(s, ()) for s, c in zip(Xb.s1_id, Xb.cand_id)), bool, len(Xb))
                loco[f"xenc+stack {src}->{tgt}"] = {"gate": round(fg, 5), "stack": round(fs, 5),
                                                    "delta": round(fs - fg, 5),
                                                    "band_auc_xenc": round(roc_auc_score(lab, Xb.xenc), 5),
                                                    "band_auc_gate": round(roc_auc_score(lab, Xb.prob), 5)}
        print("[loco]", json.dumps(loco, indent=1))
        run.log(loco=loco, timing_min=round((time.time() - t0) / 60, 1))
        v = out["val"]
        run.note(f"mini F0.5 {v['f05_macro']:.5f} (gate 0.9611) by_country {v['f05_by_country']} @t={v['threshold']}; "
                 f"stack fit on fold-1 es band pairs (valid band AUC gate {auc_v['gate']:.4f} / xenc "
                 f"{auc_v['xenc']:.4f} / stack {auc_v['stack']:.4f}); LOCO {json.dumps(loco)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--xenc-dir", default=str(paths.ART_DIR / "kaggle" / "amlc-xenc"))
    ap.add_argument("--loco-dir", default="")
    ap.add_argument("--col", default="xenc_logit")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--hypothesis", default="")
    ap.add_argument("--apply-test", default="", help="run_id of a finished xenc-gate-stack run")
    main(ap.parse_args())
