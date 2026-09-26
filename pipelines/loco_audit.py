"""Lane C / EXP-F0: LOCO-driven feature-group audit -> a transfer-safe feature set.

Loads the v1_n2 train cache + ctx features ONCE, then for every variant (the 0710 feature set minus
one group) trains the LOCO pair (India-only -> US, US-only -> India; features_v1.loco recipe:
60k source S1, lr 0.1) and scores the target country's mini S1. Reports, per variant and direction,
F0.5 at the target-tuned threshold and at the source-tuned threshold, and a paired bootstrap of the
per-S1 F0.5 against the reference variant ("full") on identical S1.

Primary statistic = LOCO-avg delta (mean of the two directions). Seed noise is measured by re-running
"full" with other seeds (--ref-seeds), which changes both the S1 subsample and LightGBM bagging.

    python pipelines/loco_audit.py --variants all --ref-seeds 43,44 --threads 8
    python pipelines/loco_audit.py --variants -B_meta,-G3_lfrac --seeds 42,43
    python pipelines/loco_audit.py --variants "combo:-B_meta+-G3_lfrac"    # drop several groups at once
"""
import _bootstrap  # noqa: F401

import argparse
import gc
import json
import time

import numpy as np
import pandas as pd
import polars as pl

import features_v1 as fv
from ber import ctx_features as cf
from ber import harness, io
from ber.decision import assign_best_s1, threshold_matches, tune_threshold
from ber.metric import paired_bootstrap, per_entity_scores
from ber.tracking import Run

# Finer groups than features_v1._PREFIX (base features split by what they measure). Every feature of
# the 0710 set belongs to exactly one group (asserted below).
GROUPS = {
    # base (cache) features
    "B_name": ["name_tset", "name_tsort", "name_ratio", "name_partial", "comp_jw", "comp_partial", "alias_tset"],
    "B_addr": ["addr_tset", "addr_ratio", "street_tset"],
    "B_tok": ["extra_tok", "missing_tok", "common_tok"],
    "B_num": ["house_rel", "num_jacc"],
    "B_legal_state": ["legal_rel", "state_rel"],
    "B_rank": ["name_rank_in_s1", "addr_rank_in_s1", "name_gap_s1", "n_cand_s1"],   # competition within the S1 list
    "B_len": ["len_core_s1", "len_core_c"],
    "B_meta": ["kmask", "rbits", "cand_kind", "cand_native", "cand_src", "cand_addr_empty"],  # provenance / format
    # ctx (ber.ctx_features v3)
    "G1_s1": ["s1_name_c", "s1_name_self", "s1_name_c_eq1", "s1_name_self_eq1", "n_key_equal"],
    "G1_pool": ["pool_name_c", "pool_name_self", "pool_name_c_eq1", "pool_name_self_eq1"],
    "G2_s1": ["s1_addr_c", "s1_addr_self", "s1_addr_c_eq1", "s1_addr_self_eq1", "s1_hs_c", "s1_hs_self",
              "s1_hs_c_eq1", "s1_hs_self_eq1", "a_key_equal", "hs_key_equal"],
    "G2_pool": ["pool_addr_c", "pool_addr_self", "pool_addr_c_eq1", "pool_addr_self_eq1"],
    "G3_lfrac": ["ex_ldf_min", "ex_ldf_max", "ex_lfrac_cmax", "ex_lfrac_cmin",
                 "mi_ldf_min", "mi_ldf_max", "mi_lfrac_cmax", "mi_lfrac_cmin"],        # token-rarity (density) typed
    "G3_edit": ["ex_n", "mi_n", "ex_unseen_n", "mi_unseen_n", "ex_biz", "mi_biz", "ex_digit", "mi_digit",
                "sub_jw", "edit_type", "s1_prefix_of_c", "c_prefix_of_s1", "first_tok_eq"],
    "G4": ["house_lev", "house_logdiff", "house_reldiff", "house_same_len", "house_first_diff",
           "house_last_diff", "sec_num_rel", "postcode_rel"],
    "G5": ["sib_n", "sib_house_agree", "sib_s1house_agree", "sib_namekey_same", "sib_house_frac",
           "sib_s1house_frac"],
}
# extra (overlapping) groups for finer drops; not part of the partition check
SUBGROUPS = {"B_retr": ["kmask", "rbits"], "B_fmt": ["cand_kind", "cand_native", "cand_src", "cand_addr_empty"],
             "G3_lfrac_raw": ["ex_lfrac_cmax", "ex_lfrac_cmin", "mi_lfrac_cmax", "mi_lfrac_cmin"],
             "G3_ldf": ["ex_ldf_min", "ex_ldf_max", "mi_ldf_min", "mi_ldf_max"]}
Q05 = {src: name for name, (src, _) in cf.DERIVED.items()}   # 'q05' part: raw lfrac -> coarsened (rescorable)
# core similarity groups are never dropped in "all" (dropping them is obviously harmful and costs 2 runs)
SCREEN = [g for g in GROUPS if g not in ("B_name", "B_addr")]


def load(threads: int):
    fv.CTX_VER = 3
    base = fv.base_feats() + ["rbits"]
    tr_pl = fv.load_cache("v1_n2", "train")
    ev_pl = fv.load_cache("v1_n2", "mini")
    ctx = harness.EvalContext.load("mini")
    ev_pl = ev_pl.filter(pl.col("s1_id").is_in(sorted(ctx.ids)))
    sctx = cf.SplitContext.build("train", 3)
    f_tr = fv.featurize_tag("v1_n2", "train", tr_pl, sctx, 8, threads)
    f_ev = fv.featurize_tag("v1_n2", "mini", ev_pl, sctx, 2, threads)
    del sctx
    new = [c for c in f_tr.columns if c not in ("s1_id", "cand_id") and fv._group_of(c) in
           ("G1", "G2", "G3", "G4", "G5")]
    tr_pl = tr_pl.join(f_tr.select(["s1_id", "cand_id"] + new), on=["s1_id", "cand_id"], how="left")
    ev_pl = ev_pl.join(f_ev.select(["s1_id", "cand_id"] + new), on=["s1_id", "cand_id"], how="left")
    feats = base + new
    grouped = sorted(f for g in GROUPS.values() for f in g)
    assert sorted(feats) == grouped, (set(feats) ^ set(grouped))
    tr_pl, ev_pl = cf.add_derived(tr_pl, list(Q05.values())), cf.add_derived(ev_pl, list(Q05.values()))
    tr = tr_pl.select(["s1_id", "cand_id", "label", "is_es"] + feats + list(Q05.values())).to_pandas()
    ev = ev_pl.select(["s1_id", "cand_id"] + feats + list(Q05.values())).to_pandas()
    folds = io.load_folds()
    cmap = dict(zip(folds.s1_id, folds.country))
    del tr_pl, ev_pl, f_tr, f_ev, folds
    gc.collect()
    return tr, ev, feats, ctx, cmap


def parse_variant(v: str, feats: list[str]) -> list[str]:
    """'full' | '-G' | 'q05' | 'combo:-G1+-G2+q05' -> feature list (q05 = raw lfrac replaced by the coarsened
    ctx_features.DERIVED columns; applied after the drops)."""
    if v == "full":
        return feats
    parts = v.split(":", 1)[1].split("+") if v.startswith("combo:") else [v]
    drop, q05 = set(), False
    for p in parts:
        if p == "q05":
            q05 = True
            continue
        g = {**GROUPS, **SUBGROUPS}
        assert p.startswith("-") and p[1:] in g, f"bad variant part {p!r}"
        drop |= set(g[p[1:]])
    out = [f for f in feats if f not in drop]
    return [Q05.get(f, f) for f in out] if q05 else out


def loco_scores(tr, ev, feats, truth, cmap, threads, n_s1, seed):
    """features_v1.loco with a seed and per-S1 scores kept (at the target-tuned threshold)."""
    cats = [c for c in fv.CAT_BASE + fv.CAT_NEW if c in feats]
    ctry_tr, ctry_ev = tr["s1_id"].map(cmap), ev["s1_id"].map(cmap)
    cs = sorted(set(ctry_ev.dropna()))
    rng = np.random.default_rng(seed)
    out, per = {}, {}
    for src in cs:
        ids = tr.loc[ctry_tr == src, "s1_id"].unique()
        ids = set(rng.choice(np.sort(ids), size=min(n_s1, len(ids)), replace=False))
        m = fv.train_lgb(tr[tr.s1_id.isin(ids)], feats, cats, threads, lr=0.1, rounds=1500, seed=seed)
        a = {}
        for c in cs:
            e = ev[ctry_ev == c]
            pr = e[["s1_id", "cand_id"]].assign(prob=m.predict(e[feats], num_threads=threads))
            a[c] = (assign_best_s1(pr), {k: v for k, v in truth.items() if cmap.get(k) == c})
        t_src, f_src, _ = tune_threshold(*a[src])
        for tgt in cs:
            if tgt == src:
                continue
            asg, tt = a[tgt]
            t_tgt, f_tgt, curve = tune_threshold(asg, tt)
            f_srct = float(curve.loc[np.isclose(curve.threshold, t_src), "f05"].iloc[0])
            key = f"{src}->{tgt}"
            out[key] = {"f05_tuned": round(f_tgt, 5), "t_tuned": t_tgt, "f05_at_src_t": round(f_srct, 5),
                        "t_src": t_src, "in_country_f05": round(f_src, 5), "best_iter": m.best_iteration}
            per[key] = per_entity_scores(threshold_matches(asg, t_tgt), tt)[["s1_id", "f05"]]
        del m
        gc.collect()
    return out, per


def main(a):
    t0 = time.time()
    tr, ev, feats, ctx, cmap = load(a.threads)
    print(f"[load] {len(tr):,} train pairs, {len(ev):,} eval pairs, {len(feats)} feats, {time.time() - t0:.0f}s",
          flush=True)
    variants = ["full"] + (["-" + g for g in SCREEN] if a.variants == "all" else
                           [v for v in a.variants.split(",") if v and v != "full"])
    seeds = [int(s) for s in a.seeds.split(",")]
    ref_seeds = [int(s) for s in a.ref_seeds.split(",") if s]
    jobs = [("full", s) for s in seeds + [s for s in ref_seeds if s not in seeds]] + \
           [(v, s) for v in variants if v != "full" for s in seeds]
    name = "loco-audit-v1-n2-ctx3" + (f"-{a.tag}" if a.tag else "")
    with Run(name, hypothesis=a.hypothesis or "LOCO feature-group audit of the 0710 feature set",
             params={"cache": "v1_n2", "ctx_ver": 3, "variants": variants, "seeds": seeds, "ref_seeds": ref_seeds,
                     "loco_n": a.loco_n, "groups": GROUPS}, tags=["features", "loco", "laneC"],
             parent="20260926-0710_aryan-bhendarkar_feat-v1-v1-n2-g1-g2-g3-g4-g5-ctx3") as run:
        res, per = {}, {}
        for v, s in jobs:
            t1 = time.time()
            fl = parse_variant(v, feats)
            out, pe = loco_scores(tr, ev, fl, ctx.truth, cmap, a.threads, a.loco_n, s)
            res[f"{v}@{s}"], per[(v, s)] = out, pe
            avg_t = np.mean([o["f05_tuned"] for o in out.values()])
            avg_s = np.mean([o["f05_at_src_t"] for o in out.values()])
            print(f"[{v}@{s}] n_feats={len(fl)} {json.dumps(out)} avg_tuned={avg_t:.5f} avg_src_t={avg_s:.5f} "
                  f"({time.time() - t1:.0f}s)", flush=True)
            pd.concat([p.assign(dir=k) for k, p in pe.items()]).to_parquet(
                run.art_dir / f"per_{v.replace(':', '_')}@{s}.parquet")
            run.log(loco_raw=res)
        # summary vs full at the same seed
        rows = []
        for (v, s), pe in per.items():
            ref = per.get(("full", s))
            r = {"variant": v, "seed": s, "n_feats": len(parse_variant(v, feats))}
            for k, o in res[f"{v}@{s}"].items():
                r[f"{k}_tuned"], r[f"{k}_src_t"] = o["f05_tuned"], o["f05_at_src_t"]
                if ref is not None and v != "full":
                    b = paired_bootstrap(ref[k], pe[k])
                    r[f"{k}_d"], r[f"{k}_ci"] = round(b["delta"], 5), [round(x, 5) for x in b["ci95"]]
                    r[f"{k}_d_src_t"] = round(o["f05_at_src_t"] - res[f"full@{s}"][k]["f05_at_src_t"], 5)
            dirs = list(res[f"{v}@{s}"])
            r["avg_tuned"] = round(float(np.mean([r[f"{k}_tuned"] for k in dirs])), 5)
            r["avg_src_t"] = round(float(np.mean([r[f"{k}_src_t"] for k in dirs])), 5)
            rows.append(r)
        summ = pd.DataFrame(rows)
        base = summ[summ.variant == "full"].set_index("seed")
        summ["d_avg_tuned"] = (summ.avg_tuned - summ.seed.map(base.avg_tuned)).round(5)
        summ["d_avg_src_t"] = (summ.avg_src_t - summ.seed.map(base.avg_src_t)).round(5)
        summ.to_csv(run.art_dir / "summary.csv", index=False)
        run.log(summary=summ.to_dict("records"), timing_min=round((time.time() - t0) / 60, 1))
        cols = ["variant", "seed", "n_feats", "avg_tuned", "d_avg_tuned", "avg_src_t", "d_avg_src_t"] + \
               [c for c in summ.columns if c.endswith("_d")]
        print(summ[cols].to_string(index=False))
        full = summ[summ.variant == "full"]
        run.note(f"LOCO audit, seeds {seeds} (+ref {ref_seeds}). full avg_tuned per seed "
                 f"{full.avg_tuned.tolist()} (seed spread {full.avg_tuned.max() - full.avg_tuned.min():.5f}).\n\n"
                 + "```\n" + summ[cols].sort_values("d_avg_tuned", ascending=False).to_string(index=False) + "\n```")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="all", help="'all' (drop each screen group) or comma list of variants")
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--ref-seeds", default="", help="extra seeds for the 'full' reference (noise estimate)")
    ap.add_argument("--loco-n", type=int, default=60_000)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--tag", default="")
    ap.add_argument("--hypothesis", default="")
    main(ap.parse_args())
