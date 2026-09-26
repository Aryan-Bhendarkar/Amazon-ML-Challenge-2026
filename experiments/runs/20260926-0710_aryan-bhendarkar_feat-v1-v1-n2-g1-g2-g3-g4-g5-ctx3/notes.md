# 20260926-0710_aryan-bhendarkar_feat-v1-v1-n2-g1-g2-g3-g4-g5-ctx3

**Hypothesis:** EXP-030 nkey_num blocking key (v1_n2) on top of ctx v3: +0.0015 mini from native-script recall

mini F0.5=0.9813 ({'India': 0.9777940414934029, 'US': 0.9836882589874294}), groups=['G1', 'G2', 'G3', 'G4', 'G5']. Top gain: ['addr_tset', 'addr_rank_in_s1', 'name_partial', 'extra_tok', 'comp_jw', 'name_ratio', 'house_rel', 'ex_n', 'kmask', 'num_jacc']


## Result (EXP-030 nkey_num blocking key by amlc-b6, v1_n2 cache; ctx v3)
- vs ctx v3 on v1_n1: mini 0.9801 → **0.9813 (+0.0012, CI [0.0010, 0.0015], p 0)**; India 0.9749 → 0.9778 (+0.0029), US flat; P 0.9963, R 0.9537, t 0.775.
- vs the pending submission model (ctx v1 on v1_n1, 0.9800): +0.0013, CI [0.0010, 0.0016], p 0.
- Blocking recall mini 0.98404 → 0.98712 (b6 run 20260926-0635_aryan-bhendarkar_blocking-nkey-num-v1-n2).
- Next: fold0x confirmation; if confirmed, replace the not-yet-uploaded sub3 (don't spend an extra submission: the gain is < 0.005).
## CONFIRMED on fold0x (run 20260926-0734_aryan-bhendarkar_confirm-fold0x-v1-n2, frozen model and t=0.775)
- fold0x 0.98117 (India 0.9782, US 0.9832) vs the pending sub3 model 0.97989: **+0.00128, CI [0.00114, 0.00142], p 0, n 353,503**. The fold0x-optimal t is 0.75 → 0.98123 (the frozen t is fine).
- Plan: test inference on v1_n2/test.parquet, then make_submission with --candidates data/cands/v1_n2/test.parquet, then an auditor delta check, then offer it to the user as the REPLACEMENT for the not-yet-uploaded sub3.

## Delta audit (audit.md): GO-with-conditions, upload INSTEAD of sub3
- No leakage, no val/test mismatch, candidates = the exact scored set (hash-verified), matches ⊆ candidates.
- **Known France caveat (I1, val cannot measure it):** nkey_num surfaces templated same-name + same-number records on DIFFERENT streets in France. About 34% of France's nkey_num-only matches are estimated FPs (street heuristic + a 20-sample eyeball). Estimated France effect −0.0008 … +0.0077 (≈ −0.0001 … +0.001 overall).
- Expected test gain vs sub3: +0.0015 … +0.003. Expected public LB ≈ 0.975–0.977 (usual −0.005 val→LB gap, B2 partly open).
- Fix next (ctx v4): name-twin competition features (best address similarity of the OTHER S1 sharing the candidate's name key).
