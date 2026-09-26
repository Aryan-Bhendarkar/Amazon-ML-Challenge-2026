"""Note 13: per-source thresholds (t_S2, t_S3) vs the global t, judged on DM-val (frozen rho/w).
Tune on DM-mini (grid), confirm frozen on DM-fold0x. Reuses saved predictions; not country-keyed.

    python pipelines/dm_persource.py --dm-ref <dm-val mini run> --mini-pred <val_pred> --fold0x-pred <fold0x_pred>
"""
import _bootstrap  # noqa: F401

import argparse
import json
import time

import numpy as np
import pandas as pd
import polars as pl

from ber import dmval, harness, io, metric, paths
from ber.tracking import Run

GRID = np.round(np.arange(0.70, 0.901, 0.025), 3)


def prep(pred_path, tag):
    ctx = harness.EvalContext.load(tag)
    ids = sorted(ctx.ids)
    p = pl.read_parquet(pred_path).filter(pl.col("s1_id").is_in(ids))
    a = p.sort("prob", descending=True).unique("cand_id", keep="first").to_pandas()
    gt = io.load_gt_pairs(ids)
    own = pd.Series(gt.s1_id.to_numpy(), index=gt.match_id.to_numpy())
    a["lab"] = own.reindex(a.cand_id).to_numpy() == a.s1_id.to_numpy()
    a["s3"] = a.cand_id.str.startswith("S3-").to_numpy()
    return a, pd.Series({k: len(ctx.truth.get(k, ())) for k in ids})


def main(x):
    t0 = time.time()
    dm = json.loads((paths.EXP_DIR / x.dm_ref / "metrics.json").read_text())["dm_params"]
    w, lo, hi, tg = dm["w"], dm["lo"], dm["hi"], dm["t_dm"]
    with Run("dm-persource", hypothesis="per-source thresholds (t_S2, t_S3) beat the global t on DM-val (note 13)",
             params=vars(x), tags=["dm-val", "decision"], parent=x.dm_ref) as run:
        a, nt = prep(x.mini_pred, "mini")
        tv = lambda t2, t3: np.where(a.s3.to_numpy(), t3, t2)  # noqa: E731
        rows = []
        for t2 in GRID:
            for t3 in GRID:
                rows.append((t2, t3, dmval.dm_entity_f05(a, nt, tv(t2, t3), w, lo, hi, x.draws).mean()))
        g = pd.DataFrame(rows, columns=["t_s2", "t_s3", "dm"])
        g.to_csv(run.art_dir / "grid_mini.csv", index=False)
        b = g.loc[g.dm.idxmax()]
        t2, t3 = float(b.t_s2), float(b.t_s3)
        print(g.sort_values("dm", ascending=False).head(8).to_string(), flush=True)
        res = {"dm_ref": x.dm_ref, "t_global": tg, "t_s2": t2, "t_s3": t3}
        for tag, path in (("mini", x.mini_pred), ("fold0x", x.fold0x_pred)):
            if tag != "mini":
                a, nt = prep(path, tag)
            ts = np.where(a.s3.to_numpy(), t3, t2)
            for name, fn in (("dm", lambda t: dmval.dm_entity_f05(a, nt, t, w, lo, hi, x.draws)),
                             ("clean", lambda t: dmval.entity_f05(a, nt, t))):
                A, B = fn(tg), fn(ts)
                res[f"{tag}_{name}"] = {"global": round(float(A.mean()), 5), "per_source": round(float(B.mean()), 5),
                                        "delta": metric.paired_bootstrap(pd.DataFrame({"s1_id": A.index, "f05": A.values}),
                                                                         pd.DataFrame({"s1_id": B.index, "f05": B.values}))}
                print(tag, name, res[f"{tag}_{name}"], flush=True)
        run.log(**res, timing_min=round((time.time() - t0) / 60, 1))
        run.note(f"t_S2 {t2} t_S3 {t3} vs global {tg}: DM-mini {res['mini_dm']['delta']['delta']:+.5f}, "
                 f"DM-fold0x {res['fold0x_dm']['delta']['delta']:+.5f}, clean fold0x {res['fold0x_clean']['delta']['delta']:+.5f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dm-ref", default="20260926-1413_aryan-bhendarkar_dm-val-mini")
    ap.add_argument("--mini-pred", default="artifacts/20260926-0710_aryan-bhendarkar_feat-v1-v1-n2-g1-g2-g3-g4-g5-ctx3/val_pred.parquet")
    ap.add_argument("--fold0x-pred", default="artifacts/20260926-1441_aryan-bhendarkar_dm-val-fold0x/fold0x_pred.parquet")
    ap.add_argument("--draws", type=int, default=5)
    main(ap.parse_args())
