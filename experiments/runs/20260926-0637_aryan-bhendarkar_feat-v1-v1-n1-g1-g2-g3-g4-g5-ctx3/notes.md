# 20260926-0637_aryan-bhendarkar_feat-v1-v1-n1-g1-g2-g3-g4-g5-ctx3

**Hypothesis:** EXP-031: ctx features on norm_v1 (transliterated names) lift India native-script pairs; +0.0002..0.0005 vs ctx v2

mini F0.5=0.9801 ({'India': 0.9749102158510577, 'US': 0.9835754013505655}), groups=['G1', 'G2', 'G3', 'G4', 'G5']. Top gain: ['addr_tset', 'addr_rank_in_s1', 'name_partial', 'extra_tok', 'name_ratio', 'comp_jw', 'rbits', 'ex_ldf_max', 'ex_n', 'house_rel']


## Result vs ctx v2 (EXP-031: ctx features on norm_v1 instead of norm_v0)
- mini 0.9800 → 0.9801 (+0.00004, CI [−0.0002, +0.0002], n.s.); India 0.9747 → 0.9749.
- LOCO: US→India 0.9230 → **0.9279 (+0.005)**: transliterated names let a US-trained model read Indian native-script names. India→US 0.9565 → 0.9552 (−0.0013, within noise).
- Keep: it's needed for the EXP-030 native-script pairs (see the v1_n2 run) and it improves the US→India transfer.
