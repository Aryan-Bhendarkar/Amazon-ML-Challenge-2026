# 20260926-1202_smoke_orphan-train-v1-n2-ctx3

**Hypothesis:** EXP-B: training on orphan-pruned views (0/19/30%, uniform+biased) makes the model robust to test's S1 removal: orphan-sim mini +0.003, clean >= -0.001, LOCO >= -0.0005

clean mini F0.5=0.9709 ({'India': 0.9637200012272256, 'US': 0.9758370962814897}); views {'view2_uniform19': {'s1_removed_total': 419377, 'view_s1_kept': 699, 'view_pairs': 66672, 'orphan_cand_pairs': 10437}, 'view3_biased19': {'s1_removed_total': 420007, 'view_s1_kept': 699, 'view_pairs': 66763, 'orphan_cand_pairs': 11390}, 'view4_uniform30': {'s1_removed_total': 662063, 'view_s1_kept': 592, 'view_pairs': 56536, 'orphan_cand_pairs': 14131}, 'view5_biased30': {'s1_removed_total': 663365, 'view_s1_kept': 527, 'view_pairs': 50225, 'orphan_cand_pairs': 13478}}. Orphan-sim eval: pipelines/orphan_sim.py (amlc-49).

