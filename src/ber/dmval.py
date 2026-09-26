"""Density-matched validation (DM-val; monitor note 10, approved 26 Sep).

Label-free finding (DIAG-ORPH/US/TWIN): test has more near-copy distractors per S1 than train. Per S1, the test
carries 2-2.6x the val mass of assigned pairs in the uncertain band. DM-val re-weights CLEAN val so that the band
negatives per S1 match test:

  rho = (test assigned pairs with p in band, per S1) / (val assigned pairs with p in band, per S1)   [one global scalar]
  w   = (rho * val_band_pairs - val_band_positives) / val_band_negatives                             [per S1 averages]
      i.e. the extra test band mass is assumed to be negatives, and positives are unchanged.

Weighting inside macro F0.5: every assigned val NEGATIVE with p in band is replicated c times, c = floor(w) +
Bernoulli(w - floor(w)) (seeded draws; results are averaged over draws). A replicated negative that is predicted
(p >= t) adds c FPs to its S1 in the per-entity F0.5 (a singleton with any FP still scores 0). Positives and
out-of-band pairs have weight 1. The band is fixed at [lo, hi] (from the shipping threshold); t is then re-tuned.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BETA2 = 0.25


def band_mass(assigned: pd.DataFrame, n_s1: int, lo: float, hi: float) -> float:
    """Assigned pairs with lo <= p <= hi per S1 (n_s1 = all S1 of the eval set, including those without rows)."""
    p = assigned["prob"].to_numpy()
    return float(((p >= lo) & (p <= hi)).sum() / n_s1)


def negative_weight(rho: float, val_band_per_s1: float, val_band_pos_per_s1: float) -> float:
    neg = val_band_per_s1 - val_band_pos_per_s1
    return float(max(1.0, (rho * val_band_per_s1 - val_band_pos_per_s1) / neg))


def copies(assigned: pd.DataFrame, w: float, lo: float, hi: float, seed: int) -> np.ndarray:
    """Replication count per assigned row: band negatives -> floor(w) + Bernoulli(frac), others 1."""
    p = assigned["prob"].to_numpy()
    band_neg = (~assigned["lab"].to_numpy()) & (p >= lo) & (p <= hi)
    fl = np.floor(w)
    extra = np.random.default_rng(seed).random(len(p)) < (w - fl)
    return np.where(band_neg, fl + extra, 1.0)


def entity_f05(assigned: pd.DataFrame, n_true: pd.Series, t, c: np.ndarray | None = None) -> pd.Series:
    """Per-entity F0.5 at threshold t with FP replication counts c (None = clean). assigned: s1_id, prob, lab (bool:
    the row's record truly belongs to its S1). n_true: index = ALL eval S1, value = #true records."""
    sel = assigned["prob"].to_numpy() >= t
    lab = assigned["lab"].to_numpy()
    cc = np.ones(len(assigned)) if c is None else c
    g = pd.DataFrame({"s1_id": assigned["s1_id"].to_numpy(), "tp": (sel & lab).astype(float),
                      "fp": (sel & ~lab) * cc}).groupby("s1_id")[["tp", "fp"]].sum()
    g = g.reindex(n_true.index, fill_value=0.0)
    tp, fp, nt = g["tp"].to_numpy(), g["fp"].to_numpy(), n_true.to_numpy().astype(float)
    fn = nt - tp
    f = np.where(tp > 0, (1 + BETA2) * tp / np.maximum((1 + BETA2) * tp + BETA2 * fn + fp, 1e-12), 0.0)
    f = np.where(nt == 0, (fp == 0).astype(float), f)
    return pd.Series(f, index=n_true.index)


def dm_entity_f05(assigned: pd.DataFrame, n_true: pd.Series, t, w: float, lo: float, hi: float,
                  draws: int = 5) -> pd.Series:
    return sum(entity_f05(assigned, n_true, t, copies(assigned, w, lo, hi, seed=42 + d)) for d in range(draws)) / draws
