# Lane C experiments (amlc-box2). 4 lines per entry: change | numbers | verdict | test cost

## C0 LOCO reference repro (20260926-1443_aryan-box2_..._locore, 21a82b9)
- change: none (0710 recipe, v1_n2 ctx3 G1–G5, features.json rebuilt from the 0710 feature_gain + cache order)
- LOCO IN→US 0.95425 / US→IN 0.92988 (avg 0.94207) = the 1337 reference exactly; seeds 43/44: 0.94243 / 0.94277
- verdict: harness trusted; seed spread 0.0007 (tuned), 0.0012 (source t)
- test cost: none

## C2 norm v2: French normalization + legal forms (21a82b9, ab36bbd; docs/france_norm_v2.md)
- change: France-keyed region/department→state, bis/ter, strip "(France)", cie/compagnie→co, et→and, frs→freres, st→saint, 5arl/5as/5asu, legal `ei`
- clean Δ 0 / LOCO Δ 0 exactly (0 changed train rows); FR label-free: strong-candidate S1 0.908→0.976 (US .961, IN .968), likely-match state missing 100%→33%
- verdict: KEEP (needs-lead sign-off: gate is LOCO-blind by design)
- test cost: norm build ≈ 6 min; refeat France ≈ 24M pairs (~15–25 min); ctx + scoring via predict_test_v1 (France rows only if split)
