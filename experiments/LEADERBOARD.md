# Experiment leaderboard (val macro F0.5)

| # | run_id | F0.5 | US | India | pair P | pair R | blk recall | cands | subset | status |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 20260925-2001_aryan-bhendarkar_feat-v1-keys-v0-g1-g2-g3-g4-g5 | **0.9209** | 0.9384 | 0.8953 | 0.9956 | 0.8395 |  |  | mini | failed |
| 2 | 20260925-1531_aryan-bhendarkar_postrules-v0 | **0.9107** | 0.9303 | 0.8821 | 0.9889 | 0.8254 |  |  | mini | done |
| 3 | 20260925-1236_aryan_baseline-v0-keys-lgbm | **0.9057** | 0.9253 | 0.8771 | 0.9856 | 0.8177 | 0.8606 | 72.1 | mini | done |
| 4 | 20260925-1621_aryan-bhendarkar_control-keys-v0 | **0.9057** | 0.9253 | 0.8771 | 0.9856 | 0.8177 | 0.8606 | 72.1 | mini | done |

## Runs without a val score

- 20260925-1534_aryan-bhendarkar_blocking-v1-micro (running) — multi-retriever union (tf-idf name / name+street / empty-addr k50 / akey / hskey / skeleto
- 20260925-1621_aryan-bhendarkar_blocking-v1-micro-n0 (done) — multi-retriever union (tf-idf name+addr / name / empty-addr k50 / akey / hskey / skeleton 
- 20260925-1627_aryan-bhendarkar_blocking-v1-micro-n0 (done) — multi-retriever union (tf-idf name+addr / name / empty-addr k50 / akey / hskey / skeleton 
- 20260925-1644_aryan-bhendarkar_blocking-v1-mini-n0 (done) — multi-retriever union (tf-idf name+addr / name / empty-addr k50 / akey / hskey / skeleton 
- 20260925-1656_aryan-bhendarkar_blocking-v1-mini-n1 (done) — multi-retriever union (tf-idf name+addr / name / empty-addr k50 / akey / hskey / skeleton 
- 20260925-1657_aryan-bhendarkar_blocking-v1-micro-n0 (done) — multi-retriever union (tf-idf name+addr / name / empty-addr k50 / akey / hskey / skeleton 
- 20260925-2015_aryan-bhendarkar_gate-blocking-v1-n1 (running) — Phase 1 gate: blocking v1 union (keys + tf_na + akey/hskey/skel/tf_empty/nkey_empty, cap 1
