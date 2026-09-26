---
name: error-buckets-baseline
description: Stable error structure of this dataset across runs (baseline v0 0.9057 -> ctx2 0.9800) - irreducible floor, name-only ambiguity, blocking gaps, what paid off and what hurt
metadata:
  type: project
---
**Run 20260925-1236 baseline v0 (mini 0.9057), 25 Sep:**
- Blocking was 75% of the loss.
- S1 name-sharing is 50% (FR test 52.5%).
- Measured S1-uniqueness rules gave +0.0044. These became the ctx G1–G5 features: +0.0189.

**Run 20260926-0426 ctx2 (mini 0.9800), 26 Sep.** Loss 0.020 = in-cand FN 0.0107 + blocking 0.0059 + FP 0.0038.
- **Irreducible floor ≈ 0.011.** Don't spend more on it:
  - Name-only records: 5,415 of 5,666 FNs have another S1 with an identical or at-least-as-close name (reverse TF-IDF over all S1). Margin add-rules hurt.
  - House edit1/near: P(true) 0.68, and the model is calibrated per house_rel.
  - Co-location with random names: set-based and fuzzy address keys don't separate.
- Oracle top-k with the current order: +0.012. This is calibration inside ambiguous cells, which is why the decision layers failed.
- The val competition artifact is real but small: owner-dominated FPs +0.0009. The optimal t does not move.
- **Reducible, still open:**
  - Blocking of native-script records (4.1% miss vs 1.4% Latin). The `nkey_num` key (sorted name key + number) recovers 938 misses (+0.0015 upper bound).
  - ctx_features.NORM_V=0 (native names untransliterated in G1/G3).
- No ID leakage (Spearman S1 id vs match id ≈ 0).

**Why:** future analyses should start by checking whether EXP-030/031 were run and whether the floor estimate still holds.
**How to apply:** re-measure only the reducible buckets. Report the floor explicitly so the team doesn't chase name-only/house noise. See [[error-analysis-method]].
