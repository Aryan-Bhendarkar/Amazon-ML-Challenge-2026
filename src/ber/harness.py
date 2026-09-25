"""Standard evaluation harness: every experiment logs the SAME things the SAME way.

    from ber import harness
    ctx = harness.EvalContext.load("mini")            # eval S1 ids, truth sets, country map
    harness.log_blocking(run, cands, ctx, n_pool)     # after candidate generation
    harness.log_predictions(run, pred, ctx)           # after scoring (s1_id, cand_id, prob)
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from . import io, metric, split
from .blocking_eval import blocking_report
from .decision import assign_best_s1, expected_f05_matches, threshold_matches, tune_threshold


@dataclass
class EvalContext:
    subset: str
    ids: set
    truth: dict
    country: dict
    folds: pd.DataFrame

    @classmethod
    def load(cls, subset: str = "mini") -> "EvalContext":
        folds = io.load_folds()
        ids = split.eval_ids(folds, subset)
        truth = io.load_gt_sets(ids)
        country = dict(zip(folds.s1_id, folds.country))
        return cls(subset, ids, truth, country, folds)


def log_blocking(run, cands: pd.DataFrame, ctx: EvalContext, n_pool: int | None = None) -> dict:
    rep = blocking_report(cands, ctx.truth, n_pool=n_pool, country=ctx.country)
    run.log(blocking=rep)
    print("[blocking]", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in rep.items()})
    return rep


def log_predictions(run, pred: pd.DataFrame, ctx: EvalContext, save_pred: bool = True,
                    also_expected_f: bool = True) -> dict:
    """pred: (s1_id, cand_id, prob) for eval S1 candidates. Tunes threshold, logs full report,
    saves per-entity scores (for scripts/compare_runs.py) and predictions to run.art_dir."""
    pred = pred[pred.s1_id.isin(ctx.ids)]
    assigned = assign_best_s1(pred)
    t, f, curve = tune_threshold(assigned, ctx.truth)
    matches = threshold_matches(assigned, t)
    rep = metric.report(matches, ctx.truth, ctx.country)
    rep["threshold"] = t
    out = {"val": rep, "subset": ctx.subset}
    if also_expected_f:
        ef = expected_f05_matches(assigned[assigned.prob >= 0.02])
        out["val_expected_f05_rule"] = metric.macro_f05(ef, ctx.truth)
    run.log(**out)
    metric.per_entity_scores(matches, ctx.truth).to_parquet(run.art_dir / "val_entity_scores.parquet")
    curve.to_csv(run.art_dir / "threshold_curve.csv", index=False)
    if save_pred:
        pred.to_parquet(run.art_dir / "val_pred.parquet")
    print(f"[val:{ctx.subset}] F0.5={rep['f05_macro']:.4f} @t={t}  P={rep['pair_precision']:.4f} "
          f"R={rep['pair_recall']:.4f}  by_country={rep.get('f05_by_country')}")
    return out
