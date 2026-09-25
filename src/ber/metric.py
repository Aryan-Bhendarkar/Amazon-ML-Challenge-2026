"""Official metric re-implementation: macro F0.5 over ALL Source-1 entities in the eval set.

Per S1 entity (from the problem statement):
  * truth empty  & pred empty      -> 1.0   (correct singleton)
  * truth empty  & pred non-empty  -> 0.0
  * truth non-empty & pred empty   -> 0.0
  * otherwise F0.5 = 1.25*P*R / (0.25*P + R)   (0 when no true positive)
Entities missing from `pred` are treated as empty predictions.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

BETA2 = 0.25


def f05_entity(pred: set, truth: set) -> float:
    if not truth:
        return 1.0 if not pred else 0.0
    if not pred:
        return 0.0
    tp = len(pred & truth)
    if tp == 0:
        return 0.0
    p = tp / len(pred)
    r = tp / len(truth)
    return (1 + BETA2) * p * r / (BETA2 * p + r)


def per_entity_scores(pred: Mapping[str, Iterable[str]], truth: Mapping[str, set]) -> pd.DataFrame:
    """One row per S1 in `truth` with tp/fp/fn/f05. `truth` defines the eval set."""
    rows = []
    for s1, t in truth.items():
        p = set(pred.get(s1, ()))
        tp = len(p & t)
        rows.append((s1, len(t), len(p), tp, len(p) - tp, len(t) - tp, f05_entity(p, t)))
    return pd.DataFrame(rows, columns=["s1_id", "n_true", "n_pred", "tp", "fp", "fn", "f05"])


def macro_f05(pred: Mapping[str, Iterable[str]], truth: Mapping[str, set]) -> float:
    if not truth:
        raise ValueError("empty eval set")
    return float(np.mean([f05_entity(set(pred.get(s1, ())), t) for s1, t in truth.items()]))


def report(pred: Mapping[str, Iterable[str]], truth: Mapping[str, set],
           country: Mapping[str, str] | None = None) -> dict:
    """Full breakdown used by every experiment. Log the whole dict as run metrics."""
    df = per_entity_scores(pred, truth)
    tp, fp, fn = df.tp.sum(), df.fp.sum(), df.fn.sum()
    single = df.n_true == 0
    out = {
        "f05_macro": float(df.f05.mean()),
        "n_entities": int(len(df)),
        "pair_precision": float(tp / (tp + fp)) if tp + fp else 1.0,
        "pair_recall": float(tp / (tp + fn)) if tp + fn else 1.0,
        "singleton_rate_true": float(single.mean()),
        "singleton_acc": float((df.loc[single, "n_pred"] == 0).mean()) if single.any() else None,
        "empty_pred_on_nonsingleton": float((df.loc[~single, "n_pred"] == 0).mean()) if (~single).any() else None,
        "entities_with_fp": float((df.fp > 0).mean()),
        "avg_pred_per_entity": float(df.n_pred.mean()),
    }
    # bucket by true match count
    b = df.n_true.clip(upper=6)
    out["f05_by_ntrue"] = {str(int(k)): float(v) for k, v in df.groupby(b).f05.mean().items()}
    if country is not None:
        c = df.s1_id.map(country)
        out["f05_by_country"] = {str(k): float(v) for k, v in df.groupby(c).f05.mean().items()}
    return out


def pairs_to_sets(df: pd.DataFrame, s1_col: str = "s1_id", id_col: str = "match_id") -> dict[str, set]:
    out: dict[str, set] = defaultdict(set)
    for s1, m in zip(df[s1_col].to_numpy(), df[id_col].to_numpy()):
        out[s1].add(m)
    return dict(out)


def paired_bootstrap(a: pd.DataFrame, b: pd.DataFrame, n_boot: int = 1000, seed: int = 42) -> dict:
    """Is run B better than run A on the SAME eval entities? Inputs: per_entity_scores() frames.
    Returns mean delta (B - A), 95% CI and P(delta <= 0). Use before keeping a change."""
    m = a[["s1_id", "f05"]].merge(b[["s1_id", "f05"]], on="s1_id", suffixes=("_a", "_b"))
    d = (m.f05_b - m.f05_a).to_numpy()
    rng = np.random.default_rng(seed)
    boots = np.concatenate([d[rng.integers(0, len(d), size=(50, len(d)))].mean(axis=1)
                            for _ in range(max(1, n_boot // 50))])   # batched: bounded memory
    return {"n": int(len(d)), "delta": float(d.mean()), "ci95": [float(np.percentile(boots, 2.5)),
            float(np.percentile(boots, 97.5))], "p_not_better": float((boots <= 0).mean())}
