# 20260925-2015_aryan-bhendarkar_gate-blocking-v1-n1

**Hypothesis:** Phase 1 gate: blocking v1 union (keys + tf_na + akey/hskey/skel/tf_empty/nkey_empty, cap 100) on norm_v1 (token map), same 28 baseline features + rbits, retrained -> mini F0.5 up vs control

mini F0.5=0.9611 by_country={'India': 0.9531, 'US': 0.9665} P=0.9868 R=0.9164 t=0.725. Top feats: ['addr_tset', 'addr_rank_in_s1', 'name_partial', 'street_tset', 'extra_tok', 'comp_jw', 'name_ratio', 'house_rel']

