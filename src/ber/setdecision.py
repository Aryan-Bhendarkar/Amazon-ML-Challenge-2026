"""Set-level decision layer (strategy_v2 P2): "how many matches does this S1 have?"

Inputs are ASSIGNED stage-1 probabilities (after decision.assign_best_s1), one row per
(s1_id, cand_id). Everything here is computed per S1 from its own candidate profile, so it is
label-free at inference time; labels are only used to build training targets on OOF probs.

  s1_profile(assigned, s1_ids)      -> one row per S1: sorted top-K probs, gaps, counts, per-source counts
  oracle_k(assigned, truth)         -> best prefix length k* per S1 under the TRUE labels (target)
  choose_k(P)                       -> k maximizing expected F0.5 under a predicted distribution over k*
  topk_matches(assigned, k)         -> {s1_id: [top-k cand ids by prob]}
  rel_features(assigned)            -> per-pair relative features for a stage-2 model
"""
from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

BETA2 = 0.25
K_PROF = 10
K_MAX = 10                       # train max matches per S1 is 11; never output more than this


def _ranked(assigned: pd.DataFrame, p_col="prob") -> pd.DataFrame:
    d = assigned.sort_values(["s1_id", p_col, "cand_id"], ascending=[True, False, True]).reset_index(drop=True)
    d["rank"] = d.groupby("s1_id").cumcount().astype(np.int16)
    return d


def s1_profile(assigned: pd.DataFrame, s1_ids, p_col="prob", k: int = K_PROF) -> pd.DataFrame:
    """Profile of an S1's assigned candidate probs. S1s without candidates get zeros."""
    d = _ranked(assigned, p_col)
    top = d[d["rank"] < k]
    wide = top.pivot(index="s1_id", columns="rank", values=p_col).reindex(columns=range(k)).fillna(0.0)
    wide.columns = [f"p{i + 1}" for i in range(k)]
    P = wide.to_numpy(np.float32)
    for i in range(k - 1):
        wide[f"gap{i + 1}"] = P[:, i] - P[:, i + 1]
    g = d.groupby("s1_id")[p_col]
    agg = pd.DataFrame({"n_assigned": g.size(), "p_sum": g.sum(),
                        **{f"n_ge{int(t * 100)}": d[p_col].ge(t).groupby(d["s1_id"]).sum()
                           for t in (0.1, 0.3, 0.5, 0.7, 0.9)}})
    if "cand_id" in d:
        s3 = d["cand_id"].str.startswith("S3")
        for name, m in (("s2", ~s3), ("s3", s3)):
            agg[f"{name}_ge50"] = (d[p_col].ge(0.5) & m).groupby(d["s1_id"]).sum()
            agg[f"{name}_max"] = d[p_col].where(m, 0.0).groupby(d["s1_id"]).max()
    out = wide.join(agg, how="outer")
    out = out.reindex(pd.Index(sorted(set(s1_ids)), name="s1_id")).fillna(0.0).astype(np.float32)
    # log P(no match) under independence (a feature; probs are not calibrated)
    out["log_p_none"] = np.log(np.clip(1 - P_full(out, k), 1e-6, 1)).sum(axis=1).astype(np.float32)
    return out


def P_full(prof: pd.DataFrame, k: int = K_PROF) -> np.ndarray:
    return prof[[f"p{i + 1}" for i in range(k)]].to_numpy(np.float64)


def prefix_f(labels_sorted: np.ndarray, n_true: int, kmax: int = K_MAX) -> np.ndarray:
    """True F0.5 of predicting the top-k (k=0..kmax) of a prob-sorted candidate list."""
    L = np.zeros(kmax, dtype=np.float64)
    m = min(kmax, len(labels_sorted))
    L[:m] = labels_sorted[:m]
    tp = np.concatenate([[0.0], np.cumsum(L)])
    k = np.arange(kmax + 1, dtype=np.float64)
    f = np.where(tp > 0, (1 + BETA2) * tp / np.maximum(BETA2 * n_true + k, 1e-9), 0.0)
    f[0] = 1.0 if n_true == 0 else 0.0
    return f


def oracle_k(assigned: pd.DataFrame, truth: Mapping[str, set], p_col="prob", kmax: int = K_MAX) -> pd.DataFrame:
    """Per S1 in truth: k* = argmax_k F(top-k) (smallest k on ties), F* and the full F(k) table."""
    d = _ranked(assigned[assigned["s1_id"].isin(truth.keys())], p_col)
    d = d[d["rank"] < kmax]
    lab = np.fromiter((c in truth[s] for s, c in zip(d["s1_id"], d["cand_id"])), bool, len(d))
    groups = dict(tuple(pd.Series(lab).groupby(d["s1_id"].to_numpy())))
    ids = list(truth.keys())
    F = np.zeros((len(ids), kmax + 1))
    for i, s in enumerate(ids):
        g = groups.get(s)
        F[i] = prefix_f(g.to_numpy() if g is not None else np.zeros(0, bool), len(truth[s]), kmax)
    k = F.argmax(axis=1)
    out = pd.DataFrame({"k_star": k.astype(np.int16), "f_star": F.max(axis=1)}, index=pd.Index(ids, name="s1_id"))
    for j in range(kmax + 1):
        out[f"F{j}"] = F[:, j].astype(np.float32)
    return out


def f_given_c(kmax: int = K_MAX) -> np.ndarray:
    """F(k | c): value of predicting top-k when the top-c are exactly the true matches (c=0..kmax)."""
    k = np.arange(kmax + 1)[None, :].astype(float)
    c = np.arange(kmax + 1)[:, None].astype(float)
    M = np.where((c > 0) & (k > 0), (1 + BETA2) * np.minimum(k, c) / np.maximum(BETA2 * c + k, 1e-9), 0.0)
    M[0, 0] = 1.0
    return M


def choose_k(Pc: np.ndarray) -> np.ndarray:
    """Pc[n, C]: predicted distribution over k* classes 0..C-1 (last class = 'C-1 or more').
    Returns k maximizing expected F0.5 under F(k|c)."""
    C = Pc.shape[1]
    M = f_given_c(C - 1)                          # [c, k]
    return (Pc @ M).argmax(axis=1).astype(np.int16)


def topk_matches(assigned: pd.DataFrame, k: pd.Series, p_col="prob", floor: float = 0.0) -> dict[str, list[str]]:
    """Top-k assigned candidates per S1 (k indexed by s1_id), optionally only those with prob >= floor."""
    d = _ranked(assigned, p_col)
    kk = d["s1_id"].map(k).fillna(0).to_numpy()
    keep = (d["rank"].to_numpy() < kk) & (d[p_col].to_numpy() >= floor)
    return d[keep].groupby("s1_id")["cand_id"].agg(list).to_dict()


def rel_features(assigned: pd.DataFrame, p_col="prob") -> pd.DataFrame:
    """Per-pair relative features within the S1 (stage 2). Keeps input columns, row order by (s1, rank)."""
    d = _ranked(assigned, p_col)
    g = d.groupby("s1_id")[p_col]
    p = d[p_col]
    d["r_rank"] = d["rank"].astype(np.float32)
    d["r_pmax"] = g.transform("max").astype(np.float32)
    d["r_ratio"] = (p / d["r_pmax"].clip(lower=1e-6)).astype(np.float32)
    d["r_prev_gap"] = (g.shift(1) - p).fillna(0.0).astype(np.float32)
    d["r_next_gap"] = (p - g.shift(-1)).fillna(p).astype(np.float32)
    d["r_n_ge50"] = p.ge(0.5).groupby(d["s1_id"]).transform("sum").astype(np.float32)
    d["r_sum_above"] = g.cumsum().astype(np.float32) - p       # mass ranked above me
    d["r_n"] = g.transform("size").astype(np.float32)
    if "cand_id" in d:
        s3 = d["cand_id"].str.startswith("S3")
        d["r_src_rank"] = d.groupby(["s1_id", s3])["rank"].rank(method="first").astype(np.float32)
    return d
