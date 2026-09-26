# 20260926-1028_aryan-bhendarkar_feat-v1-v1-n2-g1-g2-g3-g4-g5-g6-ctx4

**Hypothesis:** G6 name-twin competition (addr/street margins, strict street ratio, same-house-better-street) removes templated twin FPs (audit I1) and name-only FP1; mini >= 0.9813

mini F0.5=0.9814 ({'India': 0.9780458234922919, 'US': 0.9836629478184092}), groups=['G1', 'G2', 'G3', 'G4', 'G5', 'G6']. Top gain: ['addr_tset', 'addr_rank_in_s1', 'name_partial', 'comp_jw', 'extra_tok', 'street_tset', 'house_rel', 'name_ratio', 'ex_lfrac_cmin', 'kmask']


## Result: KILL (does not fix audit I1)
- mini 0.9813 → 0.9814 (+0.00009, CI [−0.00015, +0.00031], n.s.); R 0.9537 → 0.9560, P 0.9963 → 0.9957, t 0.725.
- Twin features get little gain (twin_addr_margin rank 21; twin_hs_better and twin_house_eq ≈ 0): the templated same-name/same-number/different-street pattern barely exists in US/India training, so the model cannot learn it.
- France test check (pipelines/france_twin_check.py, france_twin_check.json): nkey-only FR matches 14,696 → 15,030, diff-street rate 35.7% → **37.3%**. Not fixed, slightly worse.
- LOCO (v1_n2): India→US 0.9570, US→India 0.9327 (no v1_n2 base LOCO to compare against).
- Lesson: a France-only failure mode cannot be fixed with a learned feature when train has no examples of it. It needs either a structural decision rule (competition tie-break among twin S1s on test) or synthetic examples.
