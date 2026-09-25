"""Char n-gram TF-IDF retrieval (blocking). Hashing (no vocabulary dict -> low memory, stable
across processes), sublinear tf, smooth idf fit on the split's own records (unsupervised: allowed),
l2 rows, float32. Multi-view: hstack per-view l2 blocks scaled by sqrt(w) -> cosine = sum w*cos_view.

    idf = fit_idf([s1_texts, pool_texts])                  # per country, per view
    Q = transform(s1_texts, idf); P = transform(pool_texts, idf)
    i, j, s = topk(Q, P, k=30)                             # sparse_dot_topn (Apache-2.0), multithreaded
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import HashingVectorizer

N_FEATURES = 1 << 20
_HV = HashingVectorizer(analyzer="char_wb", ngram_range=(3, 3), n_features=N_FEATURES,
                        alternate_sign=False, norm=None, dtype=np.float32)
_CHUNK = 200_000


def _counts(texts) -> sp.csr_matrix:
    return _HV.transform(texts)


def _workers(n: int | None) -> int:
    return n or max(1, (os.cpu_count() or 2))


def raw_counts(texts, n_jobs: int | None = None) -> sp.csr_matrix:
    """Hashed char-3gram counts, parallel over chunks. texts: sequence of str."""
    texts = list(texts)
    parts = [texts[o:o + _CHUNK] for o in range(0, len(texts), _CHUNK)]
    if len(parts) <= 1 or _workers(n_jobs) == 1:
        mats = [_counts(p) for p in parts]
    else:
        with ProcessPoolExecutor(min(_workers(n_jobs), len(parts))) as ex:
            mats = list(ex.map(_counts, parts))
    return sp.vstack(mats, format="csr") if mats else sp.csr_matrix((0, N_FEATURES), dtype=np.float32)


def fit_idf(count_mats: list[sp.csr_matrix], max_df: float = 1.0) -> np.ndarray:
    """Smooth idf from document frequencies over the given count matrices (the split's records).
    Columns with df/N > max_df get idf 0 (dropped: bounds the top-k matmul cost)."""
    n = sum(m.shape[0] for m in count_mats)
    df = np.zeros(N_FEATURES, dtype=np.int64)
    for m in count_mats:
        df += np.bincount(m.indices, minlength=N_FEATURES)
    idf = (np.log((1 + n) / (1 + df)) + 1).astype(np.float32)
    idf[df == 0] = 0
    if max_df < 1.0:
        idf[df > max_df * n] = 0
    return idf


def weight(counts: sp.csr_matrix, idf: np.ndarray) -> sp.csr_matrix:
    """sublinear tf * idf, l2-normalized rows (all-zero rows stay zero)."""
    X = counts.copy()
    X.data = (1 + np.log(X.data)) * idf[X.indices]
    X.eliminate_zeros()
    norms = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel()).astype(np.float32)
    norms[norms == 0] = 1
    X = sp.diags(1 / norms).dot(X).tocsr()
    X.data = X.data.astype(np.float32)
    return X


def combine(views: list[sp.csr_matrix], weights: list[float]) -> sp.csr_matrix:
    """hstack l2 view blocks scaled by sqrt(w); with sum(w)=1 the dot product is sum w*cos."""
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    return sp.hstack([v * np.float32(np.sqrt(x)) for v, x in zip(views, w)], format="csr", dtype=np.float32)


def topk(Q: sp.csr_matrix, P: sp.csr_matrix, k: int, threshold: float = 0.0,
         n_threads: int | None = None, chunk: int = 50_000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Top-k rows of P for every row of Q by dot product (cosine for l2 rows).
    Returns int32 query idx, int32 pool idx, float32 score (sorted desc within query)."""
    from sparse_dot_topn import sp_matmul_topn
    PT = P.T.tocsr()
    ii, jj, ss = [], [], []
    for o in range(0, Q.shape[0], chunk):
        C = sp_matmul_topn(Q[o:o + chunk], PT, top_n=k, threshold=threshold or None, sort=True,
                           n_threads=_workers(n_threads))
        C = C.tocoo()
        ii.append((C.row + o).astype(np.int32))
        jj.append(C.col.astype(np.int32))
        ss.append(C.data.astype(np.float32))
    if not ii:
        return (np.empty(0, np.int32),) * 2 + (np.empty(0, np.float32),)
    return np.concatenate(ii), np.concatenate(jj), np.concatenate(ss)
