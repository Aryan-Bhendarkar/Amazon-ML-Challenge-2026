# 20260925-1844_aadi_model-v2-pruned-union-ctx

**Hypothesis:** blocking v1 union + learned pruner + base+ctx features -> beats 0.9107

mini F0.5=0.9785 (post-rules 0.9776) by_country={'India': 0.9731264054368545, 'US': 0.982136883671973} P=0.9957 R=0.9490 t=0.75. Top feats: ['p_prune', 'cos_na', 'r_prune', 'house_rel', 'addr_tset', 'ex_idf_min', 'house_logdiff', 'legal_rel', 'num_jacc', 'sib_house_agree']

