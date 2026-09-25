# 20260926-0354_aryan-bhendarkar_xenc-stage2-v1-n1

**Hypothesis:** CONFIRM of 20260926-0201 (killed on LOCO): frozen mini-tuned xenc stage-2 on fold0x vs shipping thr

mini: shipping thr=0.98000; control best s2_thr=0.98030 -> xenc s2_thr=0.98395; bootstrap {"n": 87952, "delta": 0.003952409176694381, "ci95": [0.00360823935532186, 0.004305484960646021], "p_not_better": 0.0} | CONFIRM fold0x (s2_thr): control 0.98016 -> xenc 0.98373 (shipping thr 0.97989); {"n": 353503, "delta": 0.0038428898146090976, "ci95": [0.003662259432460021, 0.00400972227648928], "p_not_better": 0.0}


## Result: in-domain gain CONFIRMED on fold0x; verdict unchanged (KILL on LOCO, see 20260926-0201)
Frozen mini-tuned stage-2 params on fold0x (353,503 S1; fold0x stage-1 band 285,127 pairs, xenc coverage 100%):
shipping thr 0.97989 → xenc s2_thr **0.98373** (India 0.9803, US 0.9860), paired bootstrap **+0.00384, CI [+0.0037, +0.0040]**
(control s2_thr 0.98016). So the +0.004 is real IN-DOMAIN (US/India). It is not shipped, because the France proxy (LOCO with a source-only xenc)
is −0.0015 to −0.0026. The decision to trade that risk (15% of test is France) belongs to the humans.
