# Leaderboard submissions (max 5/day, 15 total)

| sub_id | run | val F0.5 | public LB | status | note |
|---|---|---|---|---|---|
| 20260925-1357_aryan | 20260925-1236_aryan_baseline-v0-keys-lgbm | 0.9057224141218779 | 0.901 | uploaded | EXP-001 baseline v0: key blocking + rapidfuzz + LightGBM |
| 20260925-1533_aryan-bhendarkar | 20260925-1531_aryan-bhendarkar_postrules-v0 | 0.9107191366513687 | 0.904555 | uploaded | EXP-002 baseline v0 + S1-context post-rules A+dropA+B (t=0.675, floor 0.05) |
| 20260926-0416_aryan-bhendarkar | 20260925-2121_aryan-bhendarkar_feat-v1-v1-n1-g1-g2-g3-g4-g5 | 0.98 |  | prepared | sub3: blocking v1 + G1-G5 features (mini 0.9800) |
| 20260926-0953_aryan-bhendarkar | 20260926-0710_aryan-bhendarkar_feat-v1-v1-n2-g1-g2-g3-g4-g5-ctx3 | 0.9813 | 0.967 | uploaded | REPLACES sub3 (not uploaded): v1_n2 blocking (nkey_num) + ctx v3 (density-robust G3 on norm_v1); mini 0.9813, fold0x 0.98117 (+0.00128 vs sub3 model) |
| 20260926-probe-fr-empty | 20260926-0710_aryan-bhendarkar_feat-v1-v1-n2-g1-g2-g3-g4-g5-ctx3 |  | 0.834 | uploaded | DIAGNOSTIC (lead-approved, DIRECTIVES Lane A.1): 0953 predictions with all 259,452 France S1 rows emptied. F_FR = (0.967 - LB_probe)/0.15 + s_FR, s_FR ~ 0.056 +- 0.01 (train singleton prior; FR public share assumed 0.15). |
