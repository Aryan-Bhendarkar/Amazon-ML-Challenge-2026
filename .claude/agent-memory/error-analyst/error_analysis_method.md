---
name: error-analysis-method
description: Method pitfalls and fast recipes for error analysis in this repo (pandas 3 quirks, TN-less crosstabs, counterfactual gain, reducibility tests, label-free test proxies, memory blowups)
metadata:
  type: feedback
---
- Crosstabs on an error-plus-TP-sample table have NO true negatives, so they cannot estimate precision. Validate rules on ALL assigned pairs with metric.
- pandas 3 quirks:
  - `str` dtype NaN owner ids survive fillna, so use `.notna()`.
  - `groupby.apply` drops the group columns, so sample with shuffle + cumcount.
  - Interval columns can't be written to parquet.
- Counterfactual per-entity F0.5 from tp/fp/fn/n_true arrays is vectorized and instant; use it for every bucket.
- The candidate caches `data/cands/v1_n1/{mini,ctx2_mini}.parquet` hold every feature per (s1_id, cand_id), so join them and bucket on model features.
- Reducibility test: per cell, compare P(true) with the mean prob. If they match, the cell is calibrated noise. Then check whether an extra signal (reverse retrieval margin, fuzzy key) splits the cell before proposing a rule.
- Blocking-key ideas can be sized in seconds with polars exact joins against the missed pairs. Run the same join on test to get a label-free "new pairs/S1" per country, including France.
- Joins on (country, house) alone OOM on the 61 GB box (house "1" blows up). Add a street-token condition.

**Why:** a crosstab without TNs overstated house-rule precision. The NaN bug mislabelled every FP as owned. An OOM killed one run.
**How to apply:** follow these steps in every `/error-analysis`. See [[error-buckets-baseline]].
