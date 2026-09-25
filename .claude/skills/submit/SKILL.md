---
name: submit
description: Prepare a leaderboard submission from a run — preflight checks, full test inference, safe TSV writing, official validator, per-country diagnostics, auditor review, and budget logging. Human-invoked only (15 submissions total).
argument-hint: "<run_id> [note]"
disable-model-invocation: true
---

# Prepare submission for run $ARGUMENTS

## Preflight (stop and report if any fails)
!`python scripts/status.py`

1. **Budget**: at least 1 submission left today. If this would leave fewer than 2 in total before Day 3, say so explicitly.
2. **Evidence**: the run's mini F0.5 beats the best previously *submitted* run's val F0.5 (compare `submissions/records/*.json`), OR this is the first-ever baseline anchor. Otherwise recommend NOT submitting.
3. The run's code is committed (`git status` clean for `src/ pipelines/ scripts/`), so the submission is reproducible.

## Test inference
4. Run the same pipeline on the TEST split. Use the **full** S1 set and the **full** S2/S3 pools. The threshold/decision rule must be the one chosen on val, never tuned on test. Save:
   - `artifacts/<run_id>/test_candidates.parquet` (s1_id, cand_id): the exact set the model scored
   - `artifacts/<run_id>/test_matches.parquet` (s1_id, match_id): after assign_best_s1 + decision rule
5. Write, validate and diagnose:
   ```
   python scripts/make_submission.py --run <run_id> --matches artifacts/<run_id>/test_matches.parquet \
       --candidates artifacts/<run_id>/test_candidates.parquet --val-f05 <mini F0.5> --note "<what's new>"
   ```
   It must print `PASS`.
6. Diagnostics sanity check against train priors (about 5.6% empty, about 3.5 matches per S1):
   - Is France's empty rate or mean matches wildly different from US/India? (collapse or over-merge)
   - Is any country above 20% empty? Investigate before uploading.
7. Delegate a final review to the **validation-auditor** agent: leakage, threshold chosen on val, format, rules, licenses, and consistency between candidates and matches. Fix any BLOCKER.

## Hand-off to the human
8. Tell the human the exact file to upload: `submissions/files/<sub_id>/matching_results.tsv`. Only ONE teammate, on ONE laptop, logs into the portal.
9. After they report the public score: `python scripts/record_lb.py <sub_id> --score <x>`. Then add the val↔LB gap to `docs/decisions.md`. The gap trend tells us whether val is trustworthy.
