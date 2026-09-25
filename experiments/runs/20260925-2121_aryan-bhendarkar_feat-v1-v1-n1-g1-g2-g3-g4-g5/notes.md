# 20260925-2121_aryan-bhendarkar_feat-v1-v1-n1-g1-g2-g3-g4-g5

**Hypothesis:** P3/P4/P5 ctx+difference features (G1-G5) on top of v1_n1 blocking beat the gate run (0.9611)

mini F0.5=0.9800 ({'India': 0.9746621041429615, 'US': 0.9836495163907002}), groups=['G1', 'G2', 'G3', 'G4', 'G5']. Top gain: ['addr_tset', 'addr_rank_in_s1', 'name_partial', 'comp_jw', 'extra_tok', 'name_ratio', 'house_rel', 'rbits', 'street_tset', 'num_jacc']


## Result vs parent (gate 20260925-2015_aryan-bhendarkar_gate-blocking-v1-n1, same v1_n1 cache, same base FEATS+rbits, same LGBM params)
- mini F0.5 **0.9611 → 0.9800 (+0.0189, paired bootstrap CI [0.0182, 0.0195], p_not_better 0)**; India 0.9531 → 0.9747, US 0.9665 → 0.9836; P 0.9960, R 0.9521, t 0.75.
- Gain in every n_true bucket (singletons +0.026, n=1 +0.040, n≥4 +0.014). Entities with any FP 3.87% → 1.23%; empty on non-singleton 0.68% → 0.42%.
- **LOCO** (60k training S1 per country, lr 0.1; base reference = run 20260925-2211_aryan-bhendarkar_feat-v1-v1-n1-base, identical protocol):
  India→US 0.9446 → 0.9587 (+0.014; at source t +0.011); US→India 0.9072 → 0.9237 (+0.017; at source t +0.020). Transfers: the LOCO gain has the same order as the in-country gain.
- **Density-shift check** (test has fewer S1/country than train): recomputing the S1-count features after dropping non-query train S1 from the counts, same model and t: 75% kept → 0.9793 (−0.0007), 50% → 0.9779 (−0.0021). Expect ≤ ~0.002 of drift on test.
- Headroom for the decision layer: prefix oracle 0.9921 vs 0.9800. 80% of the remaining loss is FN-only entities.
- Earlier keys_v0 measurement: 0.9057 → 0.9209 (run record lost to the LOCO bug, since fixed).
## Why
Name-only and co-located records become resolvable through S1-side uniqueness (G1/G2). Typed token edits (G3) and house relations (G4) separate injected business words and ±k house numbers from noise, and sibling consensus (G5) helps too. The precision jump (FP entities /3) is the main effect.
## Next
decision_v1 (Track C) on this stage 1; confirm on v1_n1 fold0x; validation-auditor before any submission; test inference via pipelines/predict_test_v1.py.
