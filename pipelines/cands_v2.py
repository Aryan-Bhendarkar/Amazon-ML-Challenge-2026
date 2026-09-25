"""EXP cands_v2, phase A: blocking v1 retrievers for a TRAIN sample + the EVAL subset in ONE pass per
country (the TF-IDF views are built once), no per-S1 cap yet. Writes the full union with cheap
blocking features + label, so a learned pruner (phase B, pipelines/model_v2.py) picks the cap.

Queries: train sample (fold>=2 'train', fold 1 'es', seed 42) + eval subset S1 ('eval').
Pool: FULL train S2+S3 of the country. IDF over ALL train S1 + pool of the country (unlabeled).

    python pipelines/cands_v2.py --subset mini --n-train 60000 --norm-v 1
Out: data/cands/v2/blk_<tag>.parquet  (s1_id, cand_id, i, j, role, label, rbits, cos_name, cos_na, cos_addr)
"""
import _bootstrap  # noqa: F401

import argparse
import gc
import time

import numpy as np
import polars as pl
import psutil

from ber import blocking as B
from ber import io, paths, split

P = dict(k_na=30, k_name=10, k_empty=50, cap_akey=100, cap_hskey=100, cap_skel=100,
         max_df_name=0.01, max_df_addr=0.02, w_na=(0.35, 0.65))
S1_COLS = ["entity_id", "country", "n_core", "n_compact", "n_alias", "a_house", "a_street", "a_full", "a_empty"]
OUT = paths.DATA_DIR / "cands" / "v2"
SEED = 42


def rss_gb():
    return psutil.Process().memory_info().rss / 1e9


def query_roles(subset: str, n_train: int) -> pl.DataFrame:
    folds = io.load_folds()
    rng = np.random.default_rng(SEED)
    tr_pool = np.sort(folds[folds.fold >= 2].s1_id.to_numpy())
    es_pool = np.sort(folds[folds.fold == 1].s1_id.to_numpy())
    tr = rng.choice(tr_pool, size=min(n_train, len(tr_pool)), replace=False)
    es = rng.choice(es_pool, size=min(n_train // 4, len(es_pool)), replace=False)
    ev = np.sort(np.array(list(split.eval_ids(folds, subset))))
    return pl.concat([pl.DataFrame({"s1_id": tr, "role": "train"}), pl.DataFrame({"s1_id": es, "role": "es"}),
                      pl.DataFrame({"s1_id": ev, "role": "eval"})])


def run_country(ctry: str, roles: pl.DataFrame, norm_v: int, log=print) -> pl.DataFrame:
    t0 = time.time()
    s1_all = (pl.scan_parquet(B.norm_file("train", 1, norm_v)).filter(pl.col("country") == ctry)
                .select(S1_COLS).collect())
    s1_all = s1_all.join(roles.rename({"s1_id": "entity_id"}), on="entity_id", how="left")
    qrow = np.flatnonzero(s1_all["role"].is_not_null().to_numpy())
    q = s1_all[qrow]
    ptbl = B.pool_table("train", ctry, norm_v)
    pool = pl.from_arrow(ptbl.select(["entity_id", "n_core", "a_street", "a_full", "a_house", "a_empty"]))
    log(f"[{ctry}] q={q.height:,} s1_all={s1_all.height:,} pool={pool.height:,}")
    parts = {}

    def step(name, fn):
        t = time.time()
        r = fn()
        parts[name] = r.select("i", "j")
        log(f"[{ctry}]   {name}: {time.time() - t:.0f}s pairs/q {r.height / max(1, q.height):.1f} rss {rss_gb():.1f}GB")

    step("keys_v0", lambda: B.keys_v0(q.select(B.KEY_COLS).to_arrow(), ptbl))
    t = time.time()
    V = B.TfViews(s1_all, pool, P["max_df_name"], P["max_df_addr"])
    log(f"[{ctry}]   tf_build {time.time() - t:.0f}s rss {rss_gb():.1f}GB")
    Qn = V.Sn[qrow]
    Qa = V.Sa[qrow]
    Qna, Pna = V.na("s1", qrow, P["w_na"]), V.na("pool", None, P["w_na"])
    step("tf_na", lambda: B.topk_frame(Qna, Pna, P["k_na"]))
    step("tf_name", lambda: B.topk_frame(Qn, V.Pn, P["k_name"]))
    emp = np.flatnonzero(pool["a_empty"].to_numpy())
    step("tf_empty", lambda: B.topk_frame(Qn, V.Pn[emp], P["k_empty"], p_map=emp))
    step("akey", lambda: B.key_join(q.select(B.akey_expr()).to_series(), pool.select(B.akey_expr()).to_series(), P["cap_akey"]))
    step("hskey", lambda: B.key_join(q.select(B.hskey_expr()).to_series(), pool.select(B.hskey_expr()).to_series(), P["cap_hskey"]))
    step("skel", lambda: B.key_join(B.skeleton(q["n_core"]), B.skeleton(pool["n_core"]), P["cap_skel"], min_len=4))
    U = B.union(parts)
    del parts
    i, j = U["i"].to_numpy().astype(np.int64), U["j"].to_numpy().astype(np.int64)
    U = U.with_columns(pl.Series("cos_name", B.rowdot(Qn, V.Pn, i, j)),
                       pl.Series("cos_addr", B.rowdot(Qa, V.Pa, i, j)),
                       pl.Series("cos_na", B.rowdot(Qna, Pna, i, j)))
    U = U.with_columns(pl.Series("s1_id", q["entity_id"].to_numpy()[i]),
                       pl.Series("cand_id", pool["entity_id"].to_numpy()[j]),
                       pl.Series("role", q["role"].to_numpy()[i]),
                       pl.lit(ctry).alias("country"))
    log(f"[{ctry}] union {U.height:,} pairs ({U.height / q.height:.1f}/q) in {time.time() - t0:.0f}s rss {rss_gb():.1f}GB")
    qids = q["entity_id"].to_list()
    del V, Qn, Qa, Qna, Pna, pool, ptbl
    gc.collect()
    return U, qids


def main(a):
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    roles = query_roles(a.subset, a.n_train)
    for ctry in B.countries_of("train", a.norm_v):
        p = OUT / f"blk_{ctry}.parquet"
        if p.exists():
            print(f"skip {p.name}")
            continue
        U, qids = run_country(ctry, roles, a.norm_v)
        gt = pl.from_pandas(io.load_gt_pairs(set(qids))[["s1_id", "match_id"]])
        U = U.join(gt.rename({"match_id": "cand_id"}).with_columns(pl.lit(True).alias("label")),
                   on=["s1_id", "cand_id"], how="left").with_columns(pl.col("label").fill_null(False))
        U.write_parquet(p, compression="zstd")
        # union recall per role
        for role in ("train", "es", "eval"):
            ids = roles.filter(pl.col("role") == role)["s1_id"]
            g = gt.filter(pl.col("s1_id").is_in(ids.to_list()))
            hit = U.filter((pl.col("role") == role) & pl.col("label")).height
            print(f"[{ctry}] {role}: union pair recall {hit / max(1, g.height):.4f} ({hit:,}/{g.height:,})", flush=True)
        del U, gt
        gc.collect()
    print(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="mini", choices=["micro", "mini", "fold0"])
    ap.add_argument("--n-train", type=int, default=60_000)
    ap.add_argument("--norm-v", type=int, default=1)
    main(ap.parse_args())
