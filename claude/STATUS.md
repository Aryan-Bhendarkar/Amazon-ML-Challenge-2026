Updated: 26 Sep 11:50 IST (placeholder written by Cowork monitor; Claude Code overwrites the top part, keep the Monitor notes)

## Monitor notes (from Cowork, 26 Sep 11:50 IST): keep this section when overwriting STATUS.md
1. **Paths:** the `claude/` folder now exists in the repo:
   - copies of `HANDOFF.md` and `COWORK_BRIEF.md`
   - `rules_and_gotchas.md` (copy of `docs/rules.md`), `strategy.md` (copy of `docs/strategy.md`), `cloud_setup.md` (copy of `infra/aws/README.md`), `research_brief.md` (latest)

   `docs/decisions.md`, `docs/eda_findings.md` and `experiments/runs/*` stay the source of truth for past results. Write `claude/STATUS.md` and `claude/experiments.md` here.
2. **Submissions:**
   - `submissions/LOG.md` shows 1 upload on 26 Sep (20260926-0953, LB 0.967). HANDOFF says 2 left today, so follow HANDOFF unless Aryan says otherwise.
   - Before any request, record LB numbers with `scripts/record_lb.py`.
3. **EXP-A, reuse density_sim:**
   - Start from `pipelines/density_sim*`, run 20260926-0455. State in STATUS whether its `r` means the fraction of S1 kept or removed, and whether its orphan variant already recomputed the context features AND let the removed S1's records compete.
   - If it already does all of HANDOFF EXP-A steps 2–4, reuse it (add the biased mask + the md5 `|orphan` mask) instead of rewriting.
   - Report **both** uniform and biased orphan drops. If uniform < 0.004 but biased ≥ 0.006, stop and flag it under "needs human". Don't pick the variant that confirms the hypothesis.
4. **Competition in val:** document which S1s compete in `assign_best_s1` at validation time (only eval S1s, or the full pool with OOF scores). For non-eval S1 you need OOF scores from a model that never saw them. The 4-fold OOF from decision_v1 exists for the v1_n1 train tag; regenerate for v1_n2+ctx3 if needed, and state which you used.
5. **Sanity numbers to print in EXP-A:**
   - simulated records per S1 (target ~5.7)
   - share of FPs whose record's true owner was removed
   - per-country P/R
   - FP on true singletons vs FP on non-singletons vs FN
6. **bmono** (monotone constraints, EXP-H) can finish, but must not delay EXP-A. Log it in `claude/experiments.md` as EXP-H1.
7. **Any gain > +0.01 from one change gets a validation-auditor pass before you report it.**
﻿
8. **(11:55) density_sim semantics, checked in code:** `r` = fraction of the NON-query S1 **kept**. All eval S1 are kept, and the dropped S1's GT copies stay as orphans unless `--clean`.
   - Two points in the file need correcting:
     - Line 72's comment says orphans "cannot exist on test". The records/S1 evidence (5.76 test vs 4.67 train) contradicts that; fix the comment.
     - The sim appears to reuse precomputed features (`base`), so the S1-side counts were NOT recomputed after removal. EXP-A must recompute them.
   - Implied prior for EXP-A (uniform 19% removal ≈ r=0.81): about −0.0028 (linear from r=0.5 → −0.0073). If EXP-A confirms roughly −0.003, orphans explain only about a third of the −0.009 excess gap. The rest is probably France.
     - In that case, **start EXP-F step 1–2 (test-pool statistics + French normalization audit) immediately** instead of waiting for tomorrow, and log it under "needs human".
     - Still run EXP-B if the biased variant is ≥ 0.004.
