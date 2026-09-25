---
name: error-analysis-method
description: Method pitfalls for error analysis in this repo (pandas 3 str NaN, TN-less crosstabs, resource limits during test jobs, fast counterfactual)
metadata:
  type: feedback
---
- Crosstabs on an error-plus-TP-sample table have NO true negatives, so they cannot estimate precision. Always validate a rule on ALL assigned pairs (asg with prob ≥ 1e-4 keeps essentially every true pair, about 550k rows for mini) with metric.macro_f05.
- pandas 3 `str` dtype: `str(dtype)` is "str", not "string"/object. NaN owner ids survive fillna loops, so use `.notna()` checks.
- Counterfactual per-entity F0.5 from val_entity_scores (tp/fp/fn/n_true) is vectorized and instant; use it for every bucket.
- When a test job runs, the user caps analysis at 2 cores / 8 GB with nice. The whole analysis fit easily (polars filter-scan of norm parquet by entity_id).

**Why:** a crosstab without TNs overstated the precision of the house-number rules. The NaN bug mislabelled every FP as "owned".
**How to apply:** follow these steps in every `/error-analysis` run. See [[error-buckets-baseline]].
