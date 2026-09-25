"""EXP-002: S1-context post-rules (ruleA + dropA + ruleB, ber.postrules) on a parent run's saved probs.

No retraining. Val: parent val_pred -> (prob >= FLOOR, same floor the test job saved) -> assign_best_s1
-> rules at the parent's threshold -> mini F0.5 + paired bootstrap vs the parent at the same floor.
Test: parent test_pred -> same steps, with key counts over ALL TEST S1 -> test_matches.parquet.
Candidates for the submission = the parent's test_candidates.parquet (rules only re-decide scored pairs).

    python pipelines/postrules_v0.py --parent <run_id> [--test]
"""
import _bootstrap  # noqa: F401

import argparse
import json
import time

import pandas as pd

from ber import harness, metric, paths, postrules
from ber.decision import assign_best_s1, threshold_matches
from ber.tracking import Run

FLOOR = 0.05   # baseline_v0 test job kept prob >= min(0.05, t)


def decide(pred: pd.DataFrame, split: str, t: float):
    asg = assign_best_s1(pred[pred.prob >= FLOOR])
    f = postrules.context_flags(asg, split)
    return asg, f


def main(a):
    t0 = time.time()
    part = paths.ART_DIR / a.parent
    t = json.loads((part / "decision.json").read_text())["threshold"]
    ctx = harness.EvalContext.load("mini")
    with Run("postrules-v0", hypothesis="S1-context rules A+dropA+B on baseline probs (+0.0044 in error analysis)",
             params={"parent_run": a.parent, "threshold": t, "floor": FLOOR, "name_tset_max_b": postrules.NAME_TSET_MAX_B,
                     "subset": "mini"}, tags=["decision"], parent=a.parent) as run:
        pred = pd.read_parquet(part / "val_pred.parquet")
        pred = pred[pred.s1_id.isin(ctx.ids)]
        asg, f = decide(pred, "train", t)
        m0 = threshold_matches(asg, t)
        m1 = postrules.to_matches(f, t)
        s0, s1 = metric.per_entity_scores(m0, ctx.truth), metric.per_entity_scores(m1, ctx.truth)
        rep = metric.report(m1, ctx.truth, ctx.country)
        rep["threshold"] = t
        base = f.prob >= t
        counts = {k: int(v) for k, v in {"ruleA_added": (f.ruleA & ~base).sum(), "ruleB_added": (f.ruleB & ~base).sum(),
                                         "dropA_removed": (base & f.dropA_cond).sum()}.items()}
        boot = metric.paired_bootstrap(s0, s1, n_boot=1000)
        curve = {}
        for tt in [0.6, 0.65, 0.675, 0.7, 0.75]:
            curve[tt] = metric.macro_f05(postrules.to_matches(f, tt), ctx.truth)
        run.log(val=rep, subset="mini", base_same_floor=float(s0.f05.mean()), vs_parent=boot,
                rule_counts=counts, t_curve=curve)
        s1.to_parquet(run.art_dir / "val_entity_scores.parquet")
        print(f"[val] base@floor {s0.f05.mean():.5f} -> rules {s1.f05.mean():.5f}  {boot}")
        print("by country", rep.get("f05_by_country"), counts, curve)
        if a.test:
            tp = pd.read_parquet(part / "test_pred.parquet")
            asg_t, ft = decide(tp, "test", t)
            mt = postrules.to_matches(ft, t)
            base_t = ft.prob >= t
            tc = {"ruleA_added": int((ft.ruleA & ~base_t).sum()), "ruleB_added": int((ft.ruleB & ~base_t).sum()),
                  "dropA_removed": int((base_t & ft.dropA_cond).sum()),
                  "n_matches": int(sum(map(len, mt.values()))), "n_s1_nonempty": len(mt)}
            # per-country change counts (diagnostic)
            cc = ft.assign(add=(ft.ruleA | ft.ruleB) & ~base_t, drop=base_t & ft.dropA_cond) \
                   .groupby("country_c")[["add", "drop"]].sum().astype(int).to_dict()
            run.log(test_rule_counts=tc, test_rule_counts_by_country=cc)
            pd.DataFrame([(s, x) for s, xs in mt.items() for x in xs], columns=["s1_id", "match_id"]) \
              .to_parquet(run.art_dir / "test_matches.parquet")
            print("[test]", tc, cc)
        run.log(timing_min=round((time.time() - t0) / 60, 2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent", required=True)
    ap.add_argument("--test", action="store_true")
    main(ap.parse_args())
