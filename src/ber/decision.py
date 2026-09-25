"""Turning pair probabilities into final match lists.

Structural fact (verified on train): every S2/S3 record matches AT MOST ONE S1 entity.
So: (1) assign each record only to its best-scoring S1, (2) then decide per S1 which of its
assigned records to output. Two decision rules:
  * threshold      — keep assigned records with prob >= t (t tuned on val for macro F0.5)
  * expected_f05   — per S1, choose top-k maximizing expected F0.5 under independent
                     Bernoulli(prob) labels (needs CALIBRATED probs; tune a prob scale on val)
"""
from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from .metric import macro_f05


def assign_best_s1(pred: pd.DataFrame, s1_col="s1_id", id_col="cand_id", p_col="prob") -> pd.DataFrame:
    """Keep, for each candidate record, only the row of its highest-prob S1.
    Adds `margin` = prob(best) - prob(second best S1 for this record) (0 if unique) — a strong
    feature for a second-stage model too."""
    d = pred.sort_values([id_col, p_col], ascending=[True, False])
    pos = d.groupby(id_col).cumcount().to_numpy()
    second = pd.Series(d[p_col].to_numpy()[pos == 1], index=d[id_col].to_numpy()[pos == 1])
    best = d[pos == 0].copy()
    best["margin"] = best[p_col].to_numpy() - best[id_col].map(second).fillna(0.0).to_numpy()
    return best


def threshold_matches(assigned: pd.DataFrame, t: float, s1_col="s1_id", id_col="cand_id",
                      p_col="prob") -> dict[str, list[str]]:
    keep = assigned[assigned[p_col] >= t]
    return keep.groupby(s1_col)[id_col].agg(list).to_dict()


def tune_threshold(assigned: pd.DataFrame, truth: Mapping[str, set],
                   grid=None, **kw) -> tuple[float, float, pd.DataFrame]:
    """Grid-search the global threshold for macro F0.5 on the eval set.
    Returns (best_t, best_score, curve)."""
    grid = np.round(np.arange(0.05, 0.96, 0.025), 3) if grid is None else grid
    rows = [(t, macro_f05(threshold_matches(assigned, t, **kw), truth)) for t in grid]
    curve = pd.DataFrame(rows, columns=["threshold", "f05"])
    i = int(curve.f05.idxmax())
    return float(curve.threshold[i]), float(curve.f05[i]), curve


# --------------------------------------------------------------------------- expected F0.5
def _poibin(ps: np.ndarray) -> np.ndarray:
    """Poisson-binomial pmf of sum of independent Bernoulli(ps)."""
    pmf = np.zeros(len(ps) + 1)
    pmf[0] = 1.0
    for i, p in enumerate(ps):
        pmf[1:i + 2] = pmf[1:i + 2] * (1 - p) + pmf[0:i + 1] * p
        pmf[0] *= (1 - p)
    return pmf


def expected_f05_topk(probs: np.ndarray, beta2: float = 0.25) -> tuple[int, np.ndarray]:
    """Given one S1's candidate probs, return (best k, E[F0.5] for k=0..K) when predicting the
    top-k by prob. Assumes independence and no true matches outside the candidate list.
    k=0 scores 1 only if the entity is truly a singleton: E = prod(1-p).
    O(K^3) — fine for K<=30; vectorize/numba before running on 1.7M entities."""
    p = np.sort(np.asarray(probs, dtype=float))[::-1]
    K = len(p)
    ev = np.zeros(K + 1)
    ev[0] = float(np.prod(1 - p))
    for k in range(1, K + 1):
        tp = _poibin(p[:k])            # P(tp = a), a=0..k
        fn = _poibin(p[k:])            # P(fn = b), b=0..K-k
        a = np.arange(k + 1)[:, None]
        b = np.arange(K - k + 1)[None, :]
        f = np.where(a > 0, (1 + beta2) * a / (beta2 * (a + b) + k), 0.0)
        ev[k] = float((tp[:, None] * fn[None, :] * f).sum())
    return int(ev.argmax()), ev


def expected_f05_matches(assigned: pd.DataFrame, s1_col="s1_id", id_col="cand_id",
                         p_col="prob", max_k: int = 30) -> dict[str, list[str]]:
    out = {}
    for s1, g in assigned.groupby(s1_col, sort=False):
        g = g.nlargest(max_k, p_col)
        k, _ = expected_f05_topk(g[p_col].to_numpy())
        if k:
            out[s1] = g[id_col].to_numpy()[:k].tolist()
    return out
