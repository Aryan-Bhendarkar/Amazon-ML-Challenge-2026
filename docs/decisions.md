# Decision log (append-only). Format: date IST | decision | evidence (run_ids / numbers) | who

- 2026-09-25 | Validation = md5 folds over train S1; eval queries vs FULL S2/S3 pool | preserves distractor density | team
- 2026-09-25 | Assign each S2/S3 record to at most one S1 (argmax) | GT has 0 records with >1 owner (7.6M pairs) | team
- 2026-09-25 | Never use `country` as a model feature; country-agnostic features only | France is test-only (15% of S1) | team
- 2026-09-25 | Transliteration via anyascii (ISC), not unidecode (GPL) | license hygiene | team
