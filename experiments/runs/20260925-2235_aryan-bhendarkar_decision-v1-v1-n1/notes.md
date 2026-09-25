# 20260925-2235_aryan-bhendarkar_decision-v1-v1-n1

**Hypothesis:** S1-local set-level decision (k* multiclass + stage-2 relative) beats a global threshold

mini: thr=0.9800, s2_thr=0.9803, kmodel=0.9798, s2_kmodel=0.9799, iso_ef=0.9792; prefix oracle 0.9921. Chosen on mini: s2_thr.


## Result (KILL, keep the global threshold)
Stage 1 = the 0.9800 features_v1 model (mini probs); set-level models trained on 4-fold OOF (lr 0.1) of the v1_n1 train tag. All set-level inputs are S1-local (pre-assignment).
- thr 0.98000 (t=0.75) | s2_thr **0.98030** (+0.0003, CI [+0.00003, +0.00056], p 0.014) | s2_kmodel 0.97993 | kmodel 0.97981 (−0.0002) | iso_ef 0.97923 (−0.0008, p 1.0)
- Prefix oracle 0.9921; K-model accuracy on k* is 0.912, but the misses cost more than the gains.
- Why: once G1–G5 are in stage 1 the per-S1 context is already inside the pair probabilities, so the probability profile adds little. Remaining loss is mostly FN-only entities (blocking misses + low-prob true copies), which no cutoff fixes.
- The stage-2 gain is below the 0.001 keep bar and was selected on mini among 5 rules → not shipped. Revisit only if a later stage 1 (cross-encoder logit) changes calibration.
