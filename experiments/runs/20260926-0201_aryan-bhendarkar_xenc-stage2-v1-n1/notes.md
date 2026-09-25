# 20260926-0201_aryan-bhendarkar_xenc-stage2-v1-n1

**Hypothesis:** xenc_logit (cross-fitted MiniLM cross-encoder, masked to stage-1 band 0.02-0.98) as a stage-2 feature beats the shipping stage-1 global threshold (0.9800) by >0.001 on mini without hurting LOCO

mini: shipping thr=0.98000; control best s2_thr=0.98030 -> xenc s2_thr=0.98395; bootstrap {"n": 87952, "delta": 0.003952409176694381, "ci95": [0.00360823935532186, 0.004305484960646021], "p_not_better": 0.0}


## Result: KILL (fails the LOCO part of the keep bar); do not ship; no test band / test scoring
Setup: stage-1 = the 0.9800 G1–G5 model (frames from decision run 20260925-2235); xenc = MiniLM-L12-H384 (MIT), cross-fitted
(2 models on hash halves of the fold 2–4 train-role S1, each scores the other half; fold 1 / mini = mean of both);
`xenc_logit` masked to NaN outside stage-1 0.02<p<0.98 (coverage of the band: train OOF 148,444 rows 100%, mini 70,935 rows 100%).
Control and xenc arms share stage-1, K-model and isotonic; the control reproduces decision-v1 exactly (thr 0.98000, s2_thr 0.98030).

| mini (87,952 S1) | control | + xenc_logit |
|---|---|---|
| thr (shipping) | 0.98000 | 0.98000 |
| s2_thr | 0.98030 | **0.98395** (India 0.9807, US 0.9861; P 0.9974, R 0.9592) |
| s2_kmodel | 0.97993 | 0.98152 |

Paired bootstrap, xenc s2_thr vs shipping thr: **+0.00395, CI [+0.0036, +0.0043]**. `xenc_logit` is #3 in stage-2 gain.
On its own band the xenc beats stage 1 on India (AUC 0.948 vs 0.924) but not on the US (0.896 vs 0.912); the stage-2 model combines them.

**LOCO (the France proxy).** Stage 2 is trained on the source country, with rules tuned on the target. `xenc_srconly` = the target-country
logits come from an xenc trained on the SOURCE country only (amlc-xenc-loco-run). That is France's situation: stage 2 learnt to trust an in-domain xenc,
and then it meets an xenc that never saw the country.

| s2_thr F0.5 | control | xenc in-domain (optimistic) | xenc source-only (France-like) |
|---|---|---|---|
| India→US | 0.98331 (thr 0.98365) | 0.98582 | **0.98109** (−0.0026 vs thr) |
| US→India | 0.97485 | 0.98035 | **0.97332** (−0.0015) |

The xenc's standalone transfer is poor. On the stage-1 band, a US-only model scores AUC 0.63 on India (in-country 0.95), and an India-only model scores 0.72 on the US
(0.90); log loss is 3×. The cross-encoder learns country-specific surface patterns. On an unseen country the shipped rule
(s2_thr + xenc) loses 0.0015–0.0026 against the current shipping threshold, so it fails "without hurting the LOCO proxy".
Rough test mix if France behaves like LOCO: 0.85·(+0.004) + 0.15·(−0.002) ≈ +0.003 expected, but France's real behaviour is unknown
(French text, which the xenc never saw), and per-country gating is a rules judgment (country is open-set, rule 3) for the humans.

Options if revisited: (1) train the xenc with country-agnostic text augmentation (token shuffles, abbreviation swaps) and re-measure LOCO;
(2) gate the xenc by a data-derived "training coverage" signal (not a country list); human decision needed under rule 3;
(3) use the xenc only as a hard-negative veto (precision side), where the AUC loss matters less.
