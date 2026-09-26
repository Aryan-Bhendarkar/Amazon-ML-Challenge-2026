# Research log (Cowork monitor)

## 2026-09-26 11:50 IST: monitor started
- **State:** LB 0.967 (sub 20260926-0953, v1_n2+ctx3); val mini 0.9813, fold0x 0.98117. Top of LB ~0.987. 36 h to the deadline.
- **Finding (label-free):** records per S1 are 4.67 in train vs 5.76 US / 5.82 IN / 5.53 FR in test. With predicted matches per S1 unchanged at ~3.3, this implies ~19% of S1 were removed and their records kept ("orphans"). This is the main suspect for the extra −0.009 of val→LB gap.
- **Prior evidence** (density_sim, run 20260926-0455, same G1–G5 model, orphan variant): r=0.5 → 0.9735 (−0.007); r=0.2_dropunm → 0.968 (−0.012); precision fell from 0.996 to 0.990 and then 0.982. Need to confirm whether r = fraction of S1 **kept**. If so, uniform 19% removal (r=0.81) predicts only −0.002…−0.003, and the EXP-A decision may hinge on the biased variant.
- **Plan:** HANDOFF EXP-A → H (see `claude/HANDOFF.md`). Primary metric = orphan-sim fold0x; guardrails = clean + LOCO.
- **Already killed** (don't re-open): set-level decision layer (+0.0003); ctx v4 G6 twins (mini n.s.; FR diff-street 35.7 → 37.3%); cross-encoder as a feature (LOCO −).
- **Running:** bmono (monotone constraints, EXP-H item), started 11:22 IST.

## 11:55 IST: density_sim read
- r = fraction of non-query S1 kept; features were not recomputed after removal. Uniform 19% prior is about -0.003, so orphans may explain only about a third of the excess gap, and France is the likely rest. Told Claude Code to pull EXP-F steps 1-2 forward if EXP-A confirms roughly -0.003.
