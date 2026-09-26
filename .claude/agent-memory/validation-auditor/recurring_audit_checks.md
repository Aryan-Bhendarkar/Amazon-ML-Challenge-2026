---
name: recurring-audit-checks
description: Recurring risks found in submission audits (val competition mismatch, France drift, density-dependent ctx features, mini-only val_pred) — check these first
metadata:
  type: project
---

Recurring issues seen in audits (first seen 2026-09-25, LB #2 postrules-v0):

1. **Val lacks S1 competition.** assign_best_s1 on val sees only mini S1s; test sees all 1.73M. Any rule/threshold that removes or adds pairs based on "who owns this record" is mis-measured. Compare per-S1 firing rates val vs test (dropA fired 1.83% val vs 0.91% test).
   **Why:** 55% of val FPs are owned by non-mini S1s (error_analysis §3).
   **How to apply:** for every decision-rule change, compute test/val per-S1 firing rates and the val break-even true-share; flag gains that depend on removing competitor-owned records.
   **Bracket recipe (2026-09-26, `comp.py`, ~2 min):** classify the fired records by GT owner (self / none / mini_other / nonquery), then recompute F with the nonquery-owned records removed ("owner always wins"). For ctx-feat-v1 this gave val +0.0189 → oracle +0.013–0.016, and the oracle-optimal t fell to 0.475.
2. **France drift in context rules.** Compare per-country firing rates and sample France fires by eye. For example, ruleB fired 2.0% on FR against about 1% elsewhere, adding real generic French names at low prob (0.06–0.2), while US fires added synthetic random names.
   **How to apply:** always sample about 12 France firings and compare prob distributions with val.
3. **fold0\mini confirmation.** Baseline-style runs saved val_pred for mini only. Since 2026-09-26, `pipelines/confirm_v1.py` on the v1_n1 fold0x cache does a frozen-t confirmation. Use it.
4. The make_submission.py duplicate-match_id and matches-not-in-candidates checks only WARN and do not fail. Check the duplicates directly, or grep the log for WARNING.

Cheap checks that worked (≤2 cores, <15 min):
- cmp the candidate TSV against the previous submission
- re-run the validator with --check-ids (about 2 min)
- re-run postrules.context_flags on val/test (about 1.5 min each)

5. **"Too-good" blocking gains (2026-09-25, gate-blocking-v1-n1, +0.055):** these turned out legitimate. The recipe that settled it in under 5 min:
   - Bucket entities by GT coverage in the control vs new candidate cache (none/partial/full) and sum the per-entity delta per bucket.
   - Compare the oracle ceiling (perfect decisions within the candidates) delta with the realized delta. About 90% realized means the gain is consistent.
   - Compare P(label | rbits) between train and eval. Equal means no label-aware retriever.
   - Recompute any learned artifact (e.g. token_map) from folds ≥1 and check equality with the saved file.
6. **Test path missing for new blocking versions.** blocking_v1/build_cache ran for the "train" split only. Always check that a test path exists before GO.
7. **LOCO often skipped** (`loco: false` in meta) on runs that add new features. Require it before a submission.
8. **Density-dependent split statistics (2026-09-26, ctx features G1–G5).** Test S1 density differs from train:

   | Country | Test S1 vs train |
   |---|---|
   | US | 0.5× |
   | France | 0.2× |
   | India | 0.92× |

   Test pool per S1 is 5.8 vs 4.7 in train (about 2× the unmatched records per S1). Any feature that is an absolute idf (log(n/df), with unseen = log n) or a raw count drifts. The team's own density sims recomputed only part of the statistics (they left idf out), and underestimated the drift about 3×.
   **How to apply:** re-run the full-statistics density sim at r = 0.5 and 0.2 with a frozen model, in both the clean and orphan variants, and compare against the parent on the same sample. Script: `experiments/runs/20260925-2121_*/audit_scripts/dens.py` / `abl.py` / `fix.py`, about 25 min at 2 threads. Findings then: idf-only −0.0045 at r = 0.5; even a ±0.2 nat idf offset costs −0.003. The model uses idf_max as an "unseen" flag.
9. **Val→LB gap is about −0.005** (0.9057 → 0.901, 0.9107 → 0.9046). Use it when judging expected LB.

See [[validation-audit-protocol]].

10. **New retrievers surface France generic-name collisions (2026-09-26, nkey_num / v1_n2).** France test names are templated ("<City> Club/Amis/Pharmacie <legal form>"). A name+number key finds same-name records on DIFFERENT streets. Those fire at 7.4% of pairs in France vs 0.9% in India, and about 35% of the France fires are different-street FPs.
   **Why:** in US/India, name + number ≈ match, and a US↔India LOCO cannot simulate name genericity.
   **How to apply:** for every blocking change, compare the per-country firing rate of the new-retriever-only pairs (the rbits bit) on test vs val. Then run the distinctive-street heuristic on France (drop street tokens with df > 0.3% of the country's S1, then token_set_ratio < 50). Check the competitor S1 too: another S1 ≥ t that lost the record.
   Scripts: `experiments/runs/20260926-0710_*/audit_scripts/chk5-7.py`, about 10 min. The heuristic is useless on India (native-script addresses).
11. **Streamed exactness check for candidate_pairs.tsv:** an order-free 64-bit hash-sum of `s1|cand` over the TSV vs the parquet (chk3.py, <1 min). It beats relying on make_submission counts.
