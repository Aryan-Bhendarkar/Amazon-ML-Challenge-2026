Updated: 27 Sep 02:45 IST (amlc-49 HEADLESS, lead prompt 02:35)
Base: **psemb 20260926-1737**, t = 0.775 (C1 = psemb + FR norm v2 = **LB 0.968**, current best)
  - clean fold0x 0.98192, DM-fold0x 0.98101
  - LOCO 0.95383 / 0.93387
Running now: tmux xs (`logs/chainXS.sh`): xenc stack
  - **A** = psemb + `xenc_minilm` (existing cross-fitted MiniLM logits as a LightGBM feature; NaN where not scored, i.e. stage-1 p < 0.01)
  - **B** = A minus the 4 ldf features (ex/mi_ldf_min/max)
  - per variant: train + LOCO → DM-mini (frozen ρ/w, t re-tuned) → DM-fold0x (frozen t) → paired vs psemb on mini + fold0x (clean, DM, per country)
  - ETA: A ~03:40, B ~04:40 IST
  - xenc coverage/leakage checks:
    - 99.4% of train positives scored; 100% of pairs in every OOF-prob band
    - train-role OOF AUC 0.9932 = never-trained es rows 0.9934 → no in-sample leak
  - Stack choice: **feature in the GBDT** (preferred path; coverage adequate). The 3-parameter sigmoid fallback is not needed.
  - Multi-source ready: `pipelines/xenc_join.py name=dir …` → feature `xenc_<name>`. Lane G logits plug in as `features_v1 --xenc minilm,xlmr,…`.
Next: gate (DM-fold0x ≥ +0.002 vs psemb, no per-country drop, FR diagnostics) → C1-path test build (~2 h) → SUBMIT-REQUEST. Nothing will be uploaded.
Done: EXP-NL (psemb − lfrac) KILL: DM-fold0x −0.0009, LOCO direction-flip.
**Lane G handoff (box2): G3 pair set ready.**
  - `aws s3 sync s3://amlc26-699191579023/share/g3 data/kaggle/g3`
  - pairs.parquet: pair_id, tag ∈ {train, mini, fold0x, test}, cf, k1, k2; 10.76M pairs at stage-1 p ≥ 0.01
    - p ≥ 0.01 rather than 0.005, because the saved test preds are floored at 0.01
    - train cf = s1_id.hash(42) % 2 for role 'train' (= the old ckpt halves), −1 = es rows
  - records.parquet: key "<split>|<entity_id>", name (n_full), addr (a_full), house (a_house), norm_v1
  - Pair ids map back to entity ids only on box1 (data/kaggle/xenc2_map). Return logits keyed by pair_id.
**SUBMIT-REQUEST: C0 and C1** (submission #4, today's last slot; the lead picks one)
  - **C0**: `submissions/files/20260926-2034_aryan-bhendarkar/{matching_results.tsv,candidate_pairs.tsv}`
    - psemb, t = 0.775
    - validator `--check-ids` PASS; matches ⊆ candidates 5,700,127/5,700,127; 1,627,862 non-empty S1
    - clean fold0x 0.98192 (+0.00075), DM-fold0x 0.98101 (+0.0014), LOCO 0.95383 / 0.93387
  - **C1**: `submissions/files/20260926-2040_aryan-bhendarkar/…`
    - C0 + France norm v2 (24.2M France pairs re-featurized, base + ctx on norm_v2)
    - validator PASS; matches ⊆ candidates 5,704,167/5,704,167
    - **US/IN rows byte-identical to C0 (0/1,473,092 differ)**; candidate_pairs.tsv identical
    - FR: 28,101 rows differ; matches/S1 3.150 → 3.166; empty 6.30% → 6.26%; band pairs/S1 0.821 → 0.698
    - clean / DM / LOCO: identical to C0 (train unchanged under norm v2)
  - records: submissions/records/20260926-2034_aryan-bhendarkar.json, …-2040_….json
  - After upload: `python scripts/record_lb.py <sub_id> --score 0.XXX`
Next (Lane A):
  - India/word-addition discrimination ideas on DM + LOCO.
  - 23:00 overnight: nothing new needs test featurization so far (ps + emb are rescorable). Lane C's French BIZ_WORDS would need one; waiting on its merge.
**FR-probe uploaded: public 0.834 → F_France ≈ 0.944, F_non-France ≈ 0.971 (val 0.981).** The non-FR gap is the larger part (≈ 0.0086 of the 0.014).
- US test = extra band negatives (precision); India test = true copies shifted from >0.999 into the band (recall). See LB-PROBE in experiments.md.
Blockers / needs human:
  **NEEDS-LEAD: which gate governs ps?**
  - It fails the model gate (DM-fold0x +0.00036 < +0.002) but passes the transfer gate (LOCO-avg +0.003 ≥ +0.002, clean −0.00004 ≥ −0.001).
  - Its test files will be ready in `submissions/files/<sub_id>` (see `logs/ps_sub.log`; validator result logged).
  - Expected LB: small positive (≈ +0.0004 from DM, plus France transfer).
  - If you approve, it becomes submission #2 today, or the base for 21:30 with emb on top.
Submission requests:
  ~~SUBMIT-REQUEST: FR-probe~~ DONE: uploaded, public 0.834 (recorded)
  - file: `submissions/files/20260926-probe-fr-empty/matching_results.tsv` (candidate_pairs.tsv = hard link to 0953's, unchanged)
  - record: `submissions/records/20260926-probe-fr-empty.json`
  - source run: 20260926-0710 (the 0953 upload, LB 0.967)
  - change: all 259,452 France S1 rows emptied (243,522 of them were non-empty in 0953); every row and tab kept
  - validator `--check-ids`: **PASS**. 1,732,544 rows (345,976 empty, 1,386,568 non-empty); only warning = candidate file not passed (optional)
  - read-out: **F_FR = (0.967 − LB_probe) / 0.15 + s_FR**, with s_FR ≈ 0.056 ± 0.01 (train singleton prior; FR share of public-LB S1 assumed 0.15 = test share)
    - e.g. LB_probe 0.827 → F_FR ≈ 0.99; LB_probe 0.835 → F_FR ≈ 0.94
    - Sensitivity: ±0.01 on s_FR → ±0.01 on F_FR. If the FR public share is really 0.14/0.16, F_FR moves by ≈ ±0.06·(F_FR − s_FR).
  - clean / DM / LOCO: n/a (diagnostic; not a model change)
Submissions used: today 2/5 on the counter (2 left today), tomorrow 0/5

## Done since the EXP-A decision (details in .claude/experiments.md)
- DIAG-US:
  - Test's uncertain band (p 0.7–0.9) holds 2–2.6× val's mass per S1, in every country. The orphan sim reproduces only 1.2–1.4× (b19 US 1.6–1.8×).
  - Density thinning to test's S1 counts (new scenario `t19`) adds nothing (micro −0.0035 vs u19 −0.0038).
  - So the US "over-match" is uncertain-band inflation from orphan-like negatives, at a higher rate than uniform 19%.
- DIAG-FR:
  - Region↔department mismatch is in 25–30% of FR pairs but harmless: 0.007% vs 0.004% below t, median p 0.9999.
  - No norm_v2 / FR re-featurization is needed.
  - FR uncertain band = French distractor words (développement, groupe, participations, holding, cie, fils).
- EXP-E1 in-candidate sibling rescue: **KILLED** (precision 0.55–0.87 on ≤216 pairs; ΔF ≤ +0.00002).
  - The remaining recall levers are blocking misses (oracle +0.0045) and below-t FNs (+0.0101, but median p 0.34: ambiguous).

## Work split with peer sessions
- amlc-07 wrote `pipelines/orphan_train.py` (EXP-B, smoke-tested), then its session ended at about 12:15. amlc-49 now runs B0/B, the orphan_sim evals and E/F. Live peers: amlc-86, amlc-b8 (idle).

## Notes on the handoff
- `claude/rules_and_gotchas.md`, `claude/strategy.md`, `claude/cloud_setup.md` and `claude/research_brief.md` do not exist in the repo.
  I used the equivalents instead: `CLAUDE.md` (rules and cloud setup), `docs/rules.md`, `docs/strategy.md`, `docs/strategy_v2.md`,
  `docs/workflow_final.md` and `docs/decisions.md`.
- The handoff files live in `.claude/`, so STATUS.md and experiments.md are written there: `.claude/STATUS.md` and `.claude/experiments.md`.
- The box is shared with 4 other Claude sessions (amlc-b6 blocking/cache, amlc-07 ctx features/decision, amlc-86, amlc-b8).
  Only one heavy job runs at a time, and each heavy job is announced to them.

## Repo map (§0.2)
**Pipeline entry points**
- `pipelines/blocking_v1.py`, `pipelines/build_cache.py`, `pipelines/augment_nkey_num.py`: candidate caches.
  - `data/cands/v1_n2/{train,mini,fold0x,test}.parquet` hold s1_id, cand_id, label, the base pair features and rbits.
  - train = 180k S1 from folds 1–4 (17.1M pairs; role `es` marks the early-stopping S1); mini = 88k S1 / 8.4M pairs; fold0x = 353k S1 / 33.7M pairs.
- `pipelines/features_v1.py` is the current model trainer: LightGBM on base features plus `ber.ctx_features` G1–G5 (ctx v3), with an optional `--loco` step.
  - ctx features are cached as `data/cands/v1_n2/ctx3_{train,mini,fold0x}.parquet`.
- `pipelines/predict_test_v1.py`: test inference.
  - The saved full test feature matrix is `data/cands/v1_n2/test_feats_g15_ctx3.parquet` (5.7 GB).
  - `--from-feats` rescores it with any model whose features are a subset of the saved columns (minutes).
- `pipelines/confirm_v1.py`: frozen model and threshold on fold0x, plus a paired bootstrap.
- `pipelines/density_sim.py`: earlier density stress test.
  - It thins non-eval S1 only and always keeps the eval S1.
  - Its "non-clean" mode is a partial orphan sim (at r=0.5: −0.0068, clean mode −0.0022).
- **NEW** `pipelines/orphan_sim.py` + `src/ber/orphan.py` (tests: `tests/test_orphan.py`): the EXP-A/B orphan simulation.
- `scripts/make_submission.py`: writes the submission files (via `ber.submission`).

**Saved OOF scores**
- `artifacts/20260925-2235_aryan-bhendarkar_decision-v1-v1-n1/oof_train.parquet` holds 4-fold OOF stage-1 probs over the train tag, but on the OLD v1_n1 model.
- The current model has no OOF scores. mini and fold0x predictions from the current model are out-of-fold by construction, because it was trained on folds 1–4.

**Decision**
- `ber.decision.assign_best_s1`: each record keeps only its highest-prob S1 (over the S1s being scored).
- A single global threshold (0.775) is then tuned on mini by a grid search (`tune_threshold`, 0.05–0.95 in 0.025 steps) and frozen for fold0x and test.
- The expected-F / stage-2 / cardinality layers were all killed (docs/decisions.md, 26 Sep).

**How competition works at validation (the key question in §0.2)**
- The model has **no record-side competition features**: no rank within the record, no margin to the best other S1, no count of S1s that retrieved the record.
- Every feature is S1-local: its own candidate list, sibling consensus (G5), and split-level statistics.
- The only S1-pool-dependent features are the split statistics.
  - They are unsupervised counts over **ALL train S1**: G1 `s1_name_*`, G2 `s1_addr_*` / `s1_hs_*`, and G3 token df/idf over S1 core names.
  - On test they are computed over all test S1.
- Records compete only in `assign_best_s1`, and at val time only **eval S1s** (mini or fold0x) compete. The other ~96% of train S1 never compete.
  - Val is therefore already pessimistic ("owner absent") at the assignment step. The measured cost is +0.0009 FP-side (backlog EXP-017).
- Consequence for the orphan sim: removing S1 changes (a) the G1/G2/G3 split statistics and (b) which eval S1s compete in assignment.
  - `orphan_sim.py` recomputes both.
  - Full-pool competition with OOF scores for all 2.2M train S1 would need a full train featurize/score pass (~3+ h). It is not done, because the model has no competition features that would change.

## EXP-A result (details in .claude/experiments.md)
- mini: clean 0.98129 (reproduces the run exactly), u19 0.97760 (Δ −0.00374), b19 0.97739 (Δ −0.00479). The loss is about 60% FN (lfrac brittleness) and about 40% FP (297 of 1,212 FPs are orphans).

## Early EXP-A signals
- Records per S1:
  - test: FR 5.53 / IN 5.82 / US 5.76
  - train: 4.68 (both countries)
  - sim at 19% removal: 5.77, which matches.
- Label-free, current model:
  - Test predicted matches per S1 are FR 3.18 / IN 3.27 / US 3.41; val mini (predicted) is IN 3.30 / US 3.32.
  - The empty-prediction rate is 5.7–6.1% on test and 5.8–6.0% on val.
  - So there is no large excess of matches on test (US +0.09/S1 at most).
- micro smoke test (8.9k S1): clean 0.98259, uniform-19 −0.0038 [−0.0049, −0.0030], biased-19 −0.0062 [−0.0076, −0.0048].
  - The loss is mostly **recall** (R 0.954 → 0.947); precision barely moves (0.9961 → 0.9951).
  - Orphan FPs are 23 of 113 FPs.

## Monitor notes (Cowork, keep this section when overwriting STATUS.md)
**Location:** you write to `.claude/STATUS.md` / `.claude/experiments.md`, so the monitor now uses `.claude/` too. Ignore the duplicate `claude/` folder. Earlier monitor notes 1–8 are in `claude/monitor_notes.md`.

9. **(12:40 IST) EXP-A decision** (u19 −0.0037 [−0.0041, −0.0034], b19 −0.0048; records/S1 5.77 ✓; loss 60% FN / 40% FP; brittle `ex_lfrac_*` / `mi_lfrac_*`):
   - **Verdict: borderline, so take the combined path. Monitor recommendation; Aryan confirms by message in your session.** Run B0 → B, and start E and F today.
     - Orphans ≈ −0.004 of the gap.
     - The rest is France, brittle token-frequency features, and the US over-match below.
   - **The box has been IDLE since ~12:10 IST. Launch now, one heavy job at a time:**
     1. **EXP-B0, the fast candidate for submission #1:** drop the 4 lfrac features (plus a coarsened variant), retrain from the saved features, and score clean / u19 / b19 mini + LOCO.
     2. **EXP-B:** the orphan_train mixture (amlc-07 already smoke-tested it) with the winning lfrac option.
     3. **Submission #1 gate** (unchanged): u19 **fold0x** Δ ≥ +0.003 vs the 0953 model, clean ≥ −0.001, LOCO ≥ −0.0005. Rescore test from the saved test_feats (`--from-feats`) where possible.
   - **Label-free, ≤2 threads, in parallel:**
     - **US over-match:** US test predicts 3.41 matches/S1 vs ≤ 3.32 in the sim.
       - Compare set-size and top-prob histograms for US test vs US u19-val.
       - Eyeball the 20 US test S1 with the largest predicted sets and report the pattern (unit/suite/PMB variants, chain branches…).
       - Don't hand-label or tune on test.
     - **EXP-F steps 1–2:** a France token audit plus a normalization patch proposal.
   - **Ownership:**
     - amlc-07: feature code, B0, B
     - amlc-49: orphan-sim scoring, gates, STATUS / experiments
     - third session: the US/FR label-free audits

     Coordinate through STATUS; no parallel heavy jobs.

10. **(13:10 IST, monitor) DIAG-ORPH reviewed. The finding is ACCEPTED.**
   - The method holds up: the same key is used on train and test, calibration includes the 5%-FN control, and grp3+ is the clean signature. Test is **not** train with 19% of S1 removed. The extra records are near-copy distractors (singletons/pairs) that inflate the uncertain band 2–2.6×. My original orphan hypothesis was wrong at that scale. Good catch.
   - **Re-plan:**
     - **Drop EXP-B** (orphan-pruned training) and **EXP-C** (orphan sub-clusters).
     - u19/b19 stay only as a robustness guardrail (≤ −0.001 allowed), **not** the gate.
     - B0 (lfrac) continues.
   - **New primary metric: "density-matched val" (DM-val), pending Aryan's approval of item 3.** DM-val = clean val with the negative pairs in the uncertain band up-weighted so that band negatives per S1 match test.
     - Estimate one scalar ratio ρ = (test pairs in band per S1) / (val pairs in band per S1). Band = the model's p ∈ [0.2, t+0.2]. Use one global ρ; if the country split differs a lot, **report it only, never use per-country ρ**.
     - Implement it as a weight inside the macro-F computation. One option: replicate FP events by ρ in expectation, i.e. multiply each band negative's contribution to FP by ρ when it is predicted positive. Say exactly how.
     - Re-tune the global threshold t on DM-mini and confirm on DM-fold0x.
   - **Why item 3 is allowed (my view; Aryan decides):**
     - It uses a label-free test statistic (one scalar), which our rules explicitly permit ("unsupervised statistics on test are allowed").
     - HANDOFF §6 banned EM/BBSE because "orphans are invisible to them". That premise is gone. This is not EM: it is a single density ratio, it never touches test labels, and it is a global threshold.
     - Guardrails: clean fold0x Δ ≥ −0.003 (the threshold will move up, so a small clean loss is expected), LOCO not worse.
   - **Fastest test of the thesis:** re-threshold the existing 0953 test predictions at the DM-selected t (**no retraining; minutes**), plus B0 if B0 is kept by then.
     - That is **submission #1 today** if DM-fold0x Δ ≥ +0.002.
     - Expected LB direction: up, if band precision is the gap.
   - **Then:**
     1. near-copy discrimination features (house ±1 / business-word / S2∧S3-pair distractor signal: a pair of unassigned near-identical records that both beat the S1 on the other field)
     2. EXP-F France audit
     3. EXP-E recall only if it's DM-neutral (more candidates = more band distractors)

11. **(monitor) OSS/GitHub research scan triaged.** Most of it is already done or targets the orphan theory that DIAG-ORPH disproved. Three small items to add; everything else is **no**.
   - **ADD, queued after DM-val:** per-source competition features, evaluated only under DM-val (the higher band-distractor density on test is where they should pay off):
     - `rank_in_s1_same_source`, `gap_to_best_same_source` (prob or addr/name score), `n_same_source_above_0.5`
     - a record-side margin if it's computable without eval/test mismatch (test naturally has all S1; for val use OOF competitor scores, else skip)

     Note: the killed stage-2 had rank-in-S1 (not per-source) and gained +0.0003 on *clean* val. Re-judge on DM-val only.
   - **ADD to EXP-F:** hand-written French legal forms (SARL/SAS/SASU/EURL/**SCI**/SNC/SA/SCOP/SELARL, dotted variants) in the legal-form parser, so the existing legal-equal/conflict features fire on France. **No hard SCI rule** (untestable).
   - **OPTIONAL GPU track** (Kaggle is idle; only if the CPU critical path is unaffected):
     - Frozen, zero-shot `BAAI/bge-m3` (MIT, 568M; verify on the model card and log it) dense cosine for name and address, **only for pairs in the uncertain band** (test + mini + fold0x + train band).
     - Never fine-tune it.
     - Keep only if LOCO Δ ≥ 0 and DM-mini Δ ≥ +0.001.
     - Do the export and join in ≤2 threads.
   - **NO:**
     - synthetic-orphan training / has-owner classifier (disproved)
     - Splink token-frequency-product features (that is the brittle lfrac family; B0 addresses it)
     - script-ordinal features (country proxy)
     - the bge-entity-match repo, bge-reranker (same class as the LOCO-failing cross-encoder)
     - Zingg, dedupe, DeezyMatch, Unicorn, AnyMatch
   - **Integrity:** the scan listed ~11 public "Amazon ML Challenge 2026" repos. **Nobody opens them**; don't fetch or search them.

12. **(monitor) Aryan asked for an embedding signal: upgrade note 11's BGE-M3 item from optional to a PARALLEL GPU TRACK. Start it now; it must not touch the CPU critical path.**
   - **Model:** `BAAI/bge-m3` (MIT, 568M; verify the model card and log it in decisions). Fallback if it's too slow: `intfloat/multilingual-e5-small` (MIT, 118M). **Frozen, zero-shot, never fine-tuned** (the fine-tuned cross-encoder failed LOCO).
   - **Scope:** pairs in the uncertain band only, band = stage-1 p ∈ [0.05, 0.98].
     - Val/test band: the 0710 model's probs.
     - Train band: its 4-fold OOF probs (or the decision_v1 OOF). The band definition must be identical across train/val/test.
     - Outside the band the features are NaN (LightGBM handles missing).
   - **Features:** `emb_cos_name`, `emb_cos_addr`, `emb_cos_full` ("name | address"). Optionally the bge-m3 sparse lexical-match score for the name.
   - **Pipeline:**
     1. Export the unique record texts for band pairs (normalized name, address) as parquet. Use ≤2 threads on the box.
     2. Run `kaggle_gpu.py` (2×T4, fp16, max_len 64): embed records → compute pair cosines on Kaggle → return `pair_id → cos` parquet.
     3. Join and retrain. Measure the throughput on 1% first and print the ETA.
   - **Gate:** DM-mini Δ ≥ +0.001, DM-fold0x confirms, **LOCO Δ ≥ 0 in both directions**, clean ≥ −0.001. Kill it otherwise.
   - **Deadline to join the final model:** 27 Sep 12:00 IST. If it isn't validated by then, drop it.
13. **(monitor) Per-source thresholds** (t_S2, t_S3) as a decision-layer variant: tune on DM-mini, confirm on DM-fold0x, then compare to the global t. It's cheap, reuses saved preds, and is not country-keyed. Keep it only if DM-fold0x Δ ≥ +0.0005.

14. **(monitor, 14:20 IST) PARALLEL SCHEDULE until the freeze.** Three lanes, one owner each. Chain jobs in tmux queues so no lane waits on a human.
   - **Lane A: amlc-box (8 vCPU), critical path, owner amlc-49**
     - by ~15:30:
       - B0b verdict + DM-fold0x for 0710 and B0b
       - per-source thresholds (note 13, saved preds, minutes)
       - → **submission #1 today** (re-threshold/B0b) if the gate passes
     - 15:30–20:00:
       - near-copy discrimination features: per-source rank/gap (note 11), S2∧S3 decoy-pair signal, house±1/business-word interactions
       - DM-mini → DM-fold0x → LOCO
     - 20:00–22:00: the best combo → **submission #2 today** only if it clears the gate
     - **overnight** (`touch ~/.keepalive`): ONE batch test featurization of the final feature set, all new groups together, saved as test_feats_*.parquet. From then on test is rescored with `--from-feats` only.
   - **Lane B: Kaggle 2×T4, owner = a second session (low CPU)**
     - frozen bge-m3 band features (note 12): export (≤2 threads) → embed → cosines → join by 27 Sep 12:00 or drop
   - **Lane C: second box (spot r7i.2xlarge, when Aryan's AWS login is fixed), owner = third session**
     - EXP-F: France token audit, then the normalization patch (FR street types, œ/æ, St/Ste vs Sté, legal forms incl. SCI), judged by LOCO + clean-neutral
     - EXP-E: reverse retrieval + 2-hop on mini; keep only if DM-neutral
     - Deliver as feature code + a cache diff for Lane A's overnight batch
   - **27 Sep:**
     - 09:00–12:00: merge lanes, seed-bag ×5
     - 12:00: embedding cutoff
     - 14:00: final candidate
     - up to 3 submissions for final candidates (keep 2 in reserve)
     - **18:00 feature freeze**, 20:00 final rescoring + validator, 22:00 zip
   - **Rules:**
     - Only Lane A runs heavy (>4 threads) jobs on amlc-box.
     - Every result goes into `.claude/experiments.md` with DM-mini / DM-fold0x / clean / LOCO.
     - New test features happen only in the single overnight batch.

15. **(14:35 IST, monitor) France becomes the main track (agreeing with amlc-49).**
   - **Gap budget:** normal −0.005, band/twin density −0.002, leaving **≈ −0.006 unexplained**.
     - LOCO (IN→US 0.954, US→IN 0.930 vs 0.98 in-country) says an unseen country scores ~0.93–0.95. On 15% of test that is ≈ −0.006, which fits.
     - **So LOCO (the average of both directions) becomes the France proxy metric**: every Lane C change is judged by LOCO-avg Δ plus clean-neutral.
   - **Proposed (Aryan to confirm): France diagnostic as submission #1 today.**
     - File: the 0953 predictions with **all France S1 rows emptied** (keep the same candidate_pairs).
     - Why: F_FR ≈ (LB_0953 − LB_probe)/share_FR + s_FR, where s_FR = the FR singleton rate. s_FR ≈ 0.056 ± 0.01 (train prior); state the assumption. The FR share of public-LB S1 is unknown, so use 0.15 (the test share).
     - It uses today's otherwise-unused slot (no candidate passes the gate today). It is diagnosis, not model selection.
     - **Prepare it** (≤ 5 min), run the validator, and write the submission request in STATUS. **Do not upload**; Aryan decides.
   - **Lane C (box2, ready ~15:00; its own Claude session). France levers, ranked:**
     1. **LOCO-driven feature audit:**
        - For each feature group (G1–G5, base families, rbits), retrain without it on the LOCO setup; record Δ LOCO-avg and Δ clean.
        - Groups that help in-country but hurt LOCO get **coarsened or dropped** (lfrac: B0a gave IN→US +0.010 / US→IN −0.007 → try B0b q05).
        - Deliver a single "transfer-safe" feature set.
     2. **French normalization:** street types, œ/æ, St/Ste vs Sté, legal forms incl. SCI, "(France)", N°/bis/ter, CEDEX, and 5-digit postcode vs house number. Measure how many FR test pairs change (label-free) plus LOCO neutrality.
     3. **Test-pool statistics:** G1/G3 counts and IDF are computed per data-derived country on the test pool. Verify they are for France (FR has 19% shared addresses: check co-location feature distributions FR vs US/IN on test).
   - **Incident noted:** the `rm -rf experiments/runs/` glob. Recovery looks complete. **Rule:** no `rm -rf` with command substitution; delete explicit paths only.
