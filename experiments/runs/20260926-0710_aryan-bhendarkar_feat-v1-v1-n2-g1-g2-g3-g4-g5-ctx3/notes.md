# 20260926-0710_aryan-bhendarkar_feat-v1-v1-n2-g1-g2-g3-g4-g5-ctx3

**Hypothesis:** EXP-030 nkey_num blocking key (v1_n2) on top of ctx v3: +0.0015 mini from native-script recall

mini F0.5=0.9813 ({'India': 0.9777940414934029, 'US': 0.9836882589874294}), groups=['G1', 'G2', 'G3', 'G4', 'G5']. Top gain: ['addr_tset', 'addr_rank_in_s1', 'name_partial', 'extra_tok', 'comp_jw', 'name_ratio', 'house_rel', 'ex_n', 'kmask', 'num_jacc']


## Result (EXP-030 nkey_num blocking key by amlc-b6, v1_n2 cache; ctx v3)
- vs ctx v3 on v1_n1: mini 0.9801 → **0.9813 (+0.0012, CI [0.0010, 0.0015], p 0)**; India 0.9749 → 0.9778 (+0.0029), US flat; P 0.9963, R 0.9537, t 0.775.
- vs the pending submission model (ctx v1 on v1_n1, 0.9800): +0.0013, CI [0.0010, 0.0016], p 0.
- Blocking recall mini 0.98404 → 0.98712 (b6 run 20260926-0635_aryan-bhendarkar_blocking-nkey-num-v1-n2).
- Next: fold0x confirmation; if confirmed, replace the not-yet-uploaded sub3 (don't spend an extra submission: the gain is < 0.005).
