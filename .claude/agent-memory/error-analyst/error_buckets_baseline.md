---
name: error-buckets-baseline
description: Stable error structure of this dataset (first analysis, baseline v0, mini 0.9057) - name-only records, S1 name-sharing, co-location, competition bias in val; which rules paid off
metadata:
  type: project
---
Stable patterns found on 2026-09-25 (run 20260925-1236_aryan_baseline-v0-keys-lgbm, mini):
- Blocking misses were 75% of the loss (0.070 of 0.094). 2,941 of 3,439 empty-pred non-singletons had no true match in candidates.
- In-candidate loss (0.0275) is mostly calibration, not ranking: oracle top-k with the current order gave +0.0243.
- About 50% of S1 share their exact sorted core name with another S1 (FR test 52.5%). Name-only records (empty cand address, 3% of pool) are ambiguous unless the name is unique among S1. In the biggest FP bucket, 966 of 987 cases were exact feature ties with the true owner.
- Random-name TRUE matches exist at the identical S1 address (co-location). If exactly one S1 sits at that address and name_tset < 50, the pair is true about 80% of the time.
- Business words (services/center/partners/group) get injected into both true matches and distractors. House ±k / edit-1 noise likewise hits both. Both are partly irreducible.
- Measured fixes: rules using S1-side counts (ruleA exact name + unique + empty addr; dropA empty addr + name shared; ruleB same addr key + unique + name < 50) gave +0.0044 (CI 0.0041–0.0048). A top-1 fallback for empty predictions HURT at every threshold. A token_set=100 instead of exact name key HURT (−0.0049).
- Val bias: 55% of FPs were owned by non-mini S1s that never compete in val assign_best_s1.

**Why:** these recur regardless of model; the next analyses should check whether EXP-016/017 closed them.
**How to apply:** start each new analysis by re-measuring these buckets on the new run and comparing against [[error-analysis-method]].
