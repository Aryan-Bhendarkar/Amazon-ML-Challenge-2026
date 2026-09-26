# 20260926-1122_aryan-bhendarkar_feat-v1-v1-n2-g1-g2-g3-g4-g5-ctx3-mono

**Hypothesis:** monotone(+1) constraints on similarity scores: mini non-inferior to 0.9813, LOCO and France robustness not worse

mini F0.5=0.9803 ({'India': 0.9766034051100085, 'US': 0.9827552907669452}), groups=['G1', 'G2', 'G3', 'G4', 'G5']. Top gain: ['addr_tset', 'addr_rank_in_s1', 'name_partial', 'ex_ldf_max', 'extra_tok', 'house_rel', 'ex_lfrac_cmin', 'comp_jw', 'ex_n', 'name_ratio']


## Result: KILL
- mini 0.9813 → 0.9803 (**−0.0010, CI [−0.0013, −0.0008], p_not_better 1.0**); India 0.9778 → 0.9766, US 0.9837 → 0.9828.
- LOCO India→US 0.9539 / US→India 0.9313 (v1_n2 v4 run: 0.9570 / 0.9327): no transfer benefit either.
- The France slice check was stopped (one heavy job at a time, amlc-49's orphA); not needed after a clear in-country loss.
- Why: base similarity scores are legitimately non-monotone given the difference features (e.g. a very high name_tset with an extra business word is a distractor); the constraints remove that interaction.
