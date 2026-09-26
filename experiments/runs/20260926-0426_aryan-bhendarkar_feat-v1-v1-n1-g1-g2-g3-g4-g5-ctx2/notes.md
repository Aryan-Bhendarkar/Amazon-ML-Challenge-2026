# 20260926-0426_aryan-bhendarkar_feat-v1-v1-n1-g1-g2-g3-g4-g5-ctx2

**Hypothesis:** ctx v2 (density-invariant G3 token rarity) holds mini vs ctx v1 (0.9800) and cuts the density-sim loss to <=0.002 (audit B1)

mini F0.5=0.9800 ({'India': 0.9747091315769044, 'US': 0.9836523489678621}), groups=['G1', 'G2', 'G3', 'G4', 'G5']. Top gain: ['addr_tset', 'addr_rank_in_s1', 'name_partial', 'name_ratio', 'extra_tok', 'comp_jw', 'rbits', 'house_rel', 'street_tset', 'num_jacc']


## Result vs ctx v1 (20260925-2121_..., audit B1 fix)
- mini 0.9800 = 0.9800 (India 0.9747, US 0.9837; R 0.9523). LOCO: India→US 0.9565 (v1 0.9587), US→India 0.9230 (v1 0.9237); neutral within single-seed noise.
- **Density simulation** (pipelines/density_sim.py, 15k mini S1, frozen models/t; runs 20260926-0455_aryan-bhendarkar_density-sim = orphan, 20260926-0602_aryan-bhendarkar_density-sim = clean):
  clean r=0.5: v1 −0.0033, **v2 −0.0022**; clean r=0.2 (France-like): v1 −0.0046, **v2 −0.0008**; orphan scenarios tie (v2 −0.0003…+0.0009).
  v2 removes the recall collapse (R stays 0.946–0.950 vs 0.941–0.943). The residual drift is precision from the S1-count features in the orphan setting.
- Test-mix weighted (IN ~1×, US ~0.5×, FR ~0.2× density): v2 ≈ +0.0004 overall, ≈ +0.003 on France.
## Decision
KEEP v2 as the default ctx version (--ctx-ver 2) for all further work and the final build. NOT a standalone submission (expected gain ≪ 0.005).
