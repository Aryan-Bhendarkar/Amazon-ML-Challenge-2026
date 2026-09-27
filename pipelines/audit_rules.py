"""AUDIT part C: prediction-level rules on top of D (no new features). Tune on DM-mini, evaluate frozen on DM-fold0x.

    python pipelines/audit_rules.py --rules rescue,seg_src,seg_empty,seg_ncand,margin,native
Baseline selection = prob >= 0.825 on the assigned rows (D at t_DM). Every rule returns a boolean selection over the
same assigned rows (one owner per record is preserved: a rule can only add/remove a record's assignment to its best S1).
Gate (lead): DM-fold0x >= +0.0005 with paired-bootstrap CI > 0 and no per-country drop. Country is never a segment.
"""
import _bootstrap  # noqa: F401

import argparse
import itertools
import json

import numpy as np
import pandas as pd

import audit_d as A
from ber.metric import paired_bootstrap

T0 = A.DM["t"]


def top_of_s1(asg):
    """True on the highest-prob assigned row of each S1."""
    r = asg.groupby("s1_id").prob.rank(method="first", ascending=False).to_numpy()
    return r == 1


def rule_sel(name, asg, prm, cache):
    p = asg.prob.to_numpy()
    base = p >= T0
    if name == "rescue":            # S1 with an empty prediction: accept its top assigned record if p >= t2
        if "empty" not in cache:
            has = pd.Series(base).groupby(asg.s1_id.to_numpy()).transform("any").to_numpy()
            cache["empty"] = ~has
            cache["top"] = top_of_s1(asg)
        return base | (cache["empty"] & cache["top"] & (p >= prm["t2"]))
    if name == "seg_src":           # separate thresholds for S2 / S3 records
        s3 = asg.cand_src.to_numpy() == 1
        return np.where(s3, p >= prm["t_s3"], p >= prm["t_s2"])
    if name == "seg_empty":         # record with empty address
        e = asg.cand_addr_empty.to_numpy() == 1
        return np.where(e, p >= prm["t_e"], base)
    if name == "seg_ncand":         # S1 candidate-list size bucket
        n = asg.n_cand_s1.to_numpy()
        return np.where(n <= prm["n_cut"], p >= prm["t_small"], base)
    if name == "margin":            # narrow win over the 2nd S1 -> need higher p
        m = asg.margin.to_numpy()
        return base & ~((m < prm["m"]) & (p < prm["t_hi"]))
    if name == "native":            # non-Latin-script record (transliteration noise; country-agnostic)
        nat = asg.cand_native.to_numpy() == 1
        return np.where(nat, p >= prm["t_nat"], base)
    if name in ("sib", "sib_house"):  # sibling rescue: S1 already has an accepted match -> accept strong band rows
        if "has" not in cache:
            cache["has"] = pd.Series(base).groupby(asg.s1_id.to_numpy()).transform("any").to_numpy()
        ok = cache["has"] & (p >= prm["t_lo"]) & (asg.name_tset.to_numpy() >= prm["nt"]) & \
             (asg.addr_tset.to_numpy() >= prm["at"])
        if name == "sib_house":
            ok &= asg.house_rel.to_numpy() == 0
        return base | ok
    if name == "rescue+seg_src":
        s = rule_sel("seg_src", asg, prm, cache)
        if "empty2" not in cache or cache.get("k2") != (prm["t_s2"], prm["t_s3"]):
            has = pd.Series(s).groupby(asg.s1_id.to_numpy()).transform("any").to_numpy()
            cache["empty2"], cache["k2"] = ~has, (prm["t_s2"], prm["t_s3"])
            cache.setdefault("top", top_of_s1(asg))
        return s | (cache["empty2"] & cache["top"] & (p >= prm["t2"]))
    raise KeyError(name)


GRIDS = {
    "rescue": {"t2": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]},
    "seg_src": {"t_s2": [0.75, 0.8, 0.825, 0.85, 0.9], "t_s3": [0.75, 0.8, 0.825, 0.85, 0.9]},
    "seg_empty": {"t_e": [0.6, 0.7, 0.775, 0.825, 0.875, 0.925]},
    "seg_ncand": {"n_cut": [20, 50], "t_small": [0.6, 0.7, 0.775, 0.875]},
    "margin": {"m": [0.05, 0.1, 0.2, 0.3], "t_hi": [0.875, 0.925, 0.96]},
    "native": {"t_nat": [0.6, 0.7, 0.775, 0.875, 0.925]},
    "sib": {"t_lo": [0.3, 0.5, 0.65, 0.75], "nt": [90, 100], "at": [80, 95]},
    "sib_house": {"t_lo": [0.2, 0.35, 0.5, 0.65], "nt": [85, 95], "at": [70, 90]},
}


def score(asg, s1, sel, dm=True):
    return A.entity_scores(asg, s1.n_true, sel, dm=dm)


def main(a):
    mini, s1m = A.load("mini")
    fx, s1f = A.load("fold0x")
    base_m = score(mini, s1m, mini.prob.to_numpy() >= T0)
    base_f = score(fx, s1f, fx.prob.to_numpy() >= T0)
    base_fc = score(fx, s1f, fx.prob.to_numpy() >= T0, dm=False)
    print(f"[base] DM-mini {base_m.mean():.5f}  DM-fold0x {base_f.mean():.5f}  clean-fold0x {base_fc.mean():.5f}", flush=True)
    report = {"base": {"dm_mini": round(base_m.mean(), 5), "dm_fold0x": round(base_f.mean(), 5),
                       "clean_fold0x": round(base_fc.mean(), 5)}}
    cty = s1f.country
    for name in a.rules.split(","):
        grid = GRIDS[name] if name in GRIDS else {**GRIDS["seg_src"], **GRIDS["rescue"]}
        keys = list(grid)
        best, cache = None, {}
        for vals in itertools.product(*(grid[k] for k in keys)):
            prm = dict(zip(keys, vals))
            m = score(mini, s1m, rule_sel(name, mini, prm, cache)).mean()
            if best is None or m > best[0]:
                best = (m, prm)
        prm = best[1]
        cf = {}
        sf = score(fx, s1f, rule_sel(name, fx, prm, cf))
        sfc = score(fx, s1f, rule_sel(name, fx, prm, cf), dm=False)
        bs = paired_bootstrap(pd.DataFrame({"s1_id": base_f.index, "f05": base_f.to_numpy()}),
                              pd.DataFrame({"s1_id": sf.index, "f05": sf.to_numpy()}))
        bsc = paired_bootstrap(pd.DataFrame({"s1_id": base_fc.index, "f05": base_fc.to_numpy()}),
                               pd.DataFrame({"s1_id": sfc.index, "f05": sfc.to_numpy()}))
        byc = {c: round(float(sf[cty == c].mean() - base_f[cty == c].mean()), 5) for c in sorted(cty.dropna().unique())}
        passed = bs["delta"] >= 0.0005 and bs["ci95"][0] > 0 and min(byc.values()) >= 0
        r = {"params_tuned_on_dm_mini": prm, "dm_mini_delta": round(best[0] - base_m.mean(), 5),
             "dm_fold0x_delta": round(bs["delta"], 5), "dm_fold0x_ci95": [round(x, 5) for x in bs["ci95"]],
             "clean_fold0x_delta": round(bsc["delta"], 5), "clean_ci95": [round(x, 5) for x in bsc["ci95"]],
             "dm_fold0x_by_country_delta": byc, "PASS": bool(passed)}
        report[name] = r
        print(f"[{name}] {json.dumps(r)}", flush=True)
    (A.OUT / f"rules_{a.tag}.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rules", default="rescue,seg_src,seg_empty,seg_ncand,margin,native")
    ap.add_argument("--tag", default="r1")
    main(ap.parse_args())
