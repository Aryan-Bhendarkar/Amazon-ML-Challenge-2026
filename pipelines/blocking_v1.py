"""EXP-003 blocking v1: multi-retriever union (ber.blocking) — recall alone / marginal per retriever,
union recall before/after the per-S1 cap, zero-candidate non-singletons, pairs/S1, per country.

Val protocol: queries = eval-subset S1; pool = FULL train S2+S3 of the country; IDF and reverse retrieval
use ALL train S1 of the country (unlabeled). Writes the capped union to data/cands/v1/<subset>_n<norm_v>_pairs.parquet
(s1_id, cand_id, rbits, cos_name, cos_na).

    python pipelines/blocking_v1.py --subset micro     # smoke
    python pipelines/blocking_v1.py --subset mini      # gate: recall >= 0.98, zero-cand non-singletons < 0.3%
"""
import _bootstrap  # noqa: F401

import argparse
import gc
import resource
import time

import numpy as np
import polars as pl

from ber import blocking as B
from ber import harness, io, paths, split
from ber.tracking import Run

P = dict(k_name=0, k_na=30, k_empty=50, k_rev=3, cap_akey=100, cap_hskey=100, cap_skel=100, cap_nkey_empty=300, cap=100,
         max_df_name=0.01, max_df_addr=0.02, w_na=(0.35, 0.65))
S1_COLS = B.POOL_COLS             # full normalized columns (the featurizer needs them)


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def run_country(split_name, ctry, qids, log, norm_v=0, do_rev=False, keep_frames=False, retriever_frames=True):
    t0 = time.time()
    s1_all = (pl.scan_parquet(B.norm_file(split_name, 1, norm_v)).filter(pl.col("country") == ctry).select(S1_COLS)
                .collect())
    qmask = s1_all["entity_id"].is_in(list(qids)).to_numpy() if qids is not None else np.ones(s1_all.height, bool)
    qrow = np.flatnonzero(qmask)                   # query idx -> s1_all row
    q = s1_all[qrow]
    ptbl = B.pool_table(split_name, ctry, norm_v)
    pool = pl.from_arrow(ptbl.select(["entity_id", "n_core", "a_street", "a_full", "a_house", "a_empty"]))
    log(f"[{ctry}] q={q.height:,} s1_all={s1_all.height:,} pool={pool.height:,}")
    parts, tim = {}, {}

    def step(name, fn):
        t = time.time()
        r = fn()
        tim[name] = time.time() - t
        log(f"[{ctry}]   {name}: {tim[name]:.0f}s" + (f" pairs/q {r.height / max(1, q.height):.1f}" if r is not None else "")
            + f" rss {rss_gb():.1f}GB")
        if r is not None:
            parts[name] = r.select("i", "j")
        return r

    kv = step("keys_v0", lambda: B.keys_v0(q.select(B.KEY_COLS).to_arrow(), ptbl))
    V = None

    def build():
        nonlocal V
        V = B.TfViews(s1_all, pool, P["max_df_name"], P["max_df_addr"])
    step("tf_build", build)
    Qn = V.Sn[qrow]
    Qna, Pna = V.na("s1", qrow, P["w_na"]), V.na("pool", None, P["w_na"])
    step("tf_na", lambda: B.topk_frame(Qna, Pna, P["k_na"]))
    if P["k_name"]:
        step("tf_name", lambda: B.topk_frame(Qn, V.Pn, P["k_name"]))
    emp = np.flatnonzero(pool["a_empty"].to_numpy())
    step("tf_empty", lambda: B.topk_frame(Qn, V.Pn[emp], P["k_empty"], p_map=emp))
    def nkey_empty():
        r = B.key_join(q.select(B.sorted_tokens("n_core")).to_series(),
                       pool[emp].select(B.sorted_tokens("n_core")).to_series(), P["cap_nkey_empty"], min_len=3)
        return r.with_columns(pl.Series("j", emp[r["j"].to_numpy()].astype(np.uint32)))
    step("nkey_empty", nkey_empty)
    step("akey", lambda: B.key_join(q.select(B.akey_expr()).to_series(), pool.select(B.akey_expr()).to_series(), P["cap_akey"]))
    step("hskey", lambda: B.key_join(q.select(B.hskey_expr()).to_series(), pool.select(B.hskey_expr()).to_series(), P["cap_hskey"]))
    step("skel", lambda: B.key_join(B.skeleton(q["n_core"]), B.skeleton(pool["n_core"]), P["cap_skel"], min_len=4))
    if do_rev:
        def rev():
            # every pool record -> top-k over ALL S1 of the country; keep pairs whose S1 is a query
            rv = B.topk_frame(Pna, V.na("s1", None, P["w_na"]), P["k_rev"])
            s1_to_q = np.full(s1_all.height, -1, dtype=np.int64)
            s1_to_q[qrow] = np.arange(len(qrow))
            qi = s1_to_q[rv["j"].to_numpy()]
            keep = qi >= 0
            return pl.DataFrame({"i": qi[keep].astype(np.uint32), "j": rv["i"].to_numpy()[keep]})
        step("rev", rev)
    log(f"[{ctry}] retrievers done {time.time() - t0:.0f}s rss {rss_gb():.1f}GB")

    U = B.union(parts).join(kv.select("i", "j", "kmask"), on=["i", "j"], how="left") \
         .with_columns(pl.col("kmask").fill_null(0).cast(pl.Int8))
    i, j = U["i"].to_numpy().astype(np.int64), U["j"].to_numpy().astype(np.int64)
    U = U.with_columns(pl.Series("cos_name", B.rowdot(Qn, V.Pn, i, j)), pl.Series("cos_na", B.rowdot(Qna, Pna, i, j)))
    # cap score: name+address cosine; name cosine when either side has no address (max(cos_name, .) would rank
    # same-name copies at other addresses first)
    no_addr = pool["a_empty"].to_numpy()[j] | q["a_empty"].to_numpy()[i]
    U = U.with_columns(pl.Series("no_addr", no_addr))
    U = U.with_columns(pl.when(pl.col("no_addr")).then(pl.col("cos_name")).otherwise(pl.col("cos_na")).alias("score"))
    qid = q["entity_id"].to_numpy()
    pid = pool["entity_id"].to_numpy()
    fr = ({n: pl.DataFrame({"s1_id": qid[d["i"].to_numpy()], "cand_id": pid[d["j"].to_numpy()]}) for n, d in parts.items()}
          if retriever_frames else None)
    Ud = U.with_columns(pl.Series("s1_id", qid[i]), pl.Series("cand_id", pid[j]))
    del V, Qn, Qna, Pna, parts, kv
    gc.collect()
    if keep_frames:
        return fr, Ud, tim, q, ptbl
    return fr, Ud, tim


def recall(c: pl.DataFrame, gt: pl.DataFrame) -> float:
    return gt.join(c.select("s1_id", pl.col("cand_id").alias("match_id")), on=["s1_id", "match_id"], how="semi").height / max(1, gt.height)


def main(a):
    t0 = time.time()
    ctx = harness.EvalContext.load(a.subset)
    gt_all = pl.from_pandas(io.load_gt_pairs(ctx.ids)[["s1_id", "match_id"]])
    with Run(f"blocking-v1-{a.subset}-n{a.norm_v}", hypothesis="multi-retriever union (tf-idf name+addr / name / empty-addr k50 / "
             "akey / hskey / skeleton / reverse) + baseline keys -> recall >= 0.98",
             params={**P, "subset": a.subset, "norm_v": a.norm_v, "rev": a.rev}, tags=["blocking"], parent="20260925-1236_aryan_baseline-v0-keys-lgbm") as run:
        out, per = [], {}
        for ctry in B.countries_of("train"):
            ids_c = {s for s in ctx.ids if ctx.country[s] == ctry}
            if not ids_c:
                continue
            fr, U, tim = run_country("train", ctry, ids_c, print, a.norm_v, a.rev)
            gt = gt_all.filter(pl.col("s1_id").is_in(list(ids_c)))
            alone = {n: round(recall(d, gt), 4) for n, d in fr.items()}
            full = recall(U, gt)
            marg = {}
            for n in fr:
                bit = B.RBITS[n]
                rest = U.filter((pl.col("rbits") & ~pl.lit(bit)) != 0)
                marg[n] = round(full - recall(rest, gt), 4)
            capped = B.cap_per_query(U, "score", P["cap"])
            per[ctry] = dict(n_q=len(ids_c), alone=alone, marginal=marg, union_recall=round(full, 4),
                             union_pairs_per_s1=round(U.height / len(ids_c), 1),
                             capped_recall=round(recall(capped, gt), 4),
                             capped_pairs_per_s1=round(capped.height / len(ids_c), 1),
                             recall_at_cap={c: round(recall(B.cap_per_query(U, "score", c), gt), 4) for c in (50, 150, 200)},
                             pairs_per_s1={n: round(d.height / len(ids_c), 1) for n, d in fr.items()},
                             timing_s={k: round(v) for k, v in tim.items()})
            print(ctry, per[ctry])
            out.append(capped.select("s1_id", "cand_id", "rbits", "cos_name", "cos_na"))
            del fr, U
            gc.collect()
        cands = pl.concat(out)
        rep = harness.log_blocking(run, cands.select("s1_id", "cand_id").to_pandas(), ctx)
        # zero-candidate non-singletons
        hit = gt_all.join(cands.select("s1_id", pl.col("cand_id").alias("match_id")), on=["s1_id", "match_id"], how="semi")
        ns = gt_all["s1_id"].unique()
        zero = ns.len() - hit["s1_id"].n_unique()
        run.log(per_country=per, zero_cand_nonsingleton=zero / len(ctx.ids), zero_cand_nonsingleton_n=zero,
                peak_rss_gb=round(rss_gb(), 1), timing_min=round((time.time() - t0) / 60, 1))
        d = paths.ROOT / "data" / "cands" / "v1"
        d.mkdir(parents=True, exist_ok=True)
        cands.write_parquet(d / f"{a.subset}_n{a.norm_v}_pairs.parquet")
        print(f"zero-cand non-singletons {zero} ({zero / len(ctx.ids):.4%} of S1); peak rss {rss_gb():.1f}GB; "
              f"{(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="micro", choices=["micro", "mini", "fold0"])
    ap.add_argument("--norm-v", type=int, default=0)
    ap.add_argument("--rev", action="store_true", help="reverse retrieval (costs ~ a full-country forward pass)")
    main(ap.parse_args())
