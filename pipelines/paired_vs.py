"""Paired comparison of two models' saved predictions on identical S1 (clean and DM-weighted macro F0.5, overall and
per data-derived country), each at its own frozen threshold. DM weights (rho/w/band) from a dm-val run's dm_params.

    python pipelines/paired_vs.py --tag fold0x --a <pred.parquet> --ta 0.775 --b <pred.parquet> --tb 0.8 [--name ...]
"""
import _bootstrap  # noqa: F401

import argparse
import json

import numpy as np
import pandas as pd
import polars as pl

from ber import dmval, harness, io, metric, paths


def prep(path, ids):
    p = pl.read_parquet(path).filter(pl.col("s1_id").is_in(ids))
    a = p.sort("prob", descending=True).unique("cand_id", keep="first").to_pandas()
    gt = io.load_gt_pairs(ids)
    own = pd.Series(gt.s1_id.to_numpy(), index=gt.match_id.to_numpy())
    a["lab"] = own.reindex(a.cand_id).to_numpy() == a.s1_id.to_numpy()
    return a


def boot(A, B):
    return metric.paired_bootstrap(pd.DataFrame({"s1_id": A.index, "f05": A.values}),
                                   pd.DataFrame({"s1_id": B.index, "f05": B.values}))


def main(x):
    dm = json.loads((paths.EXP_DIR / x.dm_ref / "metrics.json").read_text())["dm_params"]
    ctx = harness.EvalContext.load(x.tag)
    ids = sorted(ctx.ids)
    nt = pd.Series({k: len(ctx.truth.get(k, ())) for k in ids})
    cty = pd.Series(ctx.country).reindex(ids)
    a, b = prep(x.a, ids), prep(x.b, ids)
    out = {"tag": x.tag, "a": x.a, "ta": x.ta, "b": x.b, "tb": x.tb, "dm_ref": x.dm_ref}
    for name, fa, fb in (("dm", dmval.dm_entity_f05(a, nt, x.ta, dm["w"], dm["lo"], dm["hi"]),
                          dmval.dm_entity_f05(b, nt, x.tb, dm["w"], dm["lo"], dm["hi"])),
                         ("clean", dmval.entity_f05(a, nt, x.ta), dmval.entity_f05(b, nt, x.tb))):
        r = {"a": round(float(fa.mean()), 5), "b": round(float(fb.mean()), 5), "delta": boot(fa, fb)}
        for c in sorted(cty.dropna().unique()):
            m = (cty == c).to_numpy()
            r[c] = {"a": round(float(fa[m].mean()), 5), "b": round(float(fb[m].mean()), 5), "delta": boot(fa[m], fb[m])}
        out[name] = r
        print(f"[{x.name}] {x.tag} {name}: {r['a']} -> {r['b']}  d {r['delta']['delta']:+.5f} {np.round(r['delta']['ci95'], 5).tolist()} | "
              + " ".join(f"{c} {r[c]['delta']['delta']:+.5f} {np.round(r[c]['delta']['ci95'], 5).tolist()}" for c in sorted(cty.dropna().unique())),
              flush=True)
    if x.out:
        (paths.ROOT / x.out).write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="fold0x")
    ap.add_argument("--a", required=True)
    ap.add_argument("--ta", type=float, required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--tb", type=float, required=True)
    ap.add_argument("--dm-ref", default="20260926-1413_aryan-bhendarkar_dm-val-mini")
    ap.add_argument("--name", default="")
    ap.add_argument("--out", default="")
    main(ap.parse_args())
