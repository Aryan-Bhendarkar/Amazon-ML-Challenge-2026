# Decision log (append-only). Format: date IST | decision | evidence (run_ids / numbers) | who

- 2026-09-25 | Validation = md5 folds over train S1; eval queries vs FULL S2/S3 pool | preserves distractor density | team
- 2026-09-25 | Assign each S2/S3 record to at most one S1 (argmax) | GT has 0 records with >1 owner (7.6M pairs) | team
- 2026-09-25 | Never use `country` as a model feature; country-agnostic features only | France is test-only (15% of S1) | team
- 2026-09-25 | Transliteration via anyascii (ISC), not unidecode (GPL) | license hygiene | team
- 2026-09-25 | Library `sparse-dot-topn` 1.2.0 for TF-IDF top-k blocking | license Apache-2.0 (PyPI classifier "OSI Approved :: Apache Software License"; ing-bank/sparse_dot_topn) | amlc-86
- 2026-09-25 | Learned native-script token map (NORM_VERSION 1), built from folds 1–4 GT pairs only and applied only to non-Latin-script names | holdout fold 1: token acc 0.464→0.990, name tset≥90 0.136→0.985; the v1 S1 cache is byte-identical to v0 (Latin untouched) | amlc-86
- 2026-09-25 | Session ownership: amlc-b6 owns blocking v1 + the candidate cache (data/cands/<ver>/{train,<subset>}.parquet, role column); amlc-86 owns train_eval + the token map/normalization; amlc-07 owns ctx features / decision | avoid file collisions (a blocking.py overwrite happened once) | user
