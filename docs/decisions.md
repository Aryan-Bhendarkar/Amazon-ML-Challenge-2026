# Decision log (append-only). Format: date IST | decision | evidence (run_ids / numbers) | who

- 2026-09-25 | Validation = md5 folds over train S1; eval queries vs FULL S2/S3 pool | preserves distractor density | team
- 2026-09-25 | Assign each S2/S3 record to at most one S1 (argmax) | GT has 0 records with >1 owner (7.6M pairs) | team
- 2026-09-25 | Never use `country` as a model feature; country-agnostic features only | France is test-only (15% of S1) | team
- 2026-09-25 | Transliteration via anyascii (ISC), not unidecode (GPL) | license hygiene | team
- 2026-09-25 | Library `sparse-dot-topn` 1.2.0 for TF-IDF top-k blocking | license Apache-2.0 (PyPI classifier "OSI Approved :: Apache Software License"; ing-bank/sparse_dot_topn) | amlc-86
- 2026-09-25 | Learned native-script token map (NORM_VERSION 1), built from folds 1–4 GT pairs only and applied only to non-Latin-script names | holdout fold 1: token acc 0.464→0.990, name tset≥90 0.136→0.985; the v1 S1 cache is byte-identical to v0 (Latin untouched) | amlc-86
- 2026-09-25 | Session ownership: amlc-b6 owns blocking v1 + the candidate cache (data/cands/<ver>/{train,<subset>}.parquet, role column); amlc-86 owns train_eval + the token map/normalization; amlc-07 owns ctx features / decision | avoid file collisions (a blocking.py overwrite happened once) | user
- 2026-09-25 | Pipeline v2 replaces baseline: blocking v1 retrievers (keys_v0, tf_na k30, tf_empty k50, akey, hskey, skel; tf_name dropped) -> learned LightGBM pruner on cheap blocking feats (top-40 per S1) -> baseline + blocking + ctx G1-G5 features -> LightGBM, t=0.75 | mini F0.5 0.9782 (IN 0.9728, US 0.9820) vs 0.9107 (postrules-v0); cand recall 0.861 -> 0.979, pair P 0.9956; tf_name found only 6/298k mini positives alone (-0.0003 F0.5 when dropped, ~40% less TF-IDF time). Post-rules on top: -0.0008 (features subsume them). Runs 20260925-1844 / 20260925-1857 | aadi
