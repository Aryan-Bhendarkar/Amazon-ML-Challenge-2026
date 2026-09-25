# Master brief for Claude Code: from baseline to leaderboard top

You are the team's senior ML engineer for the Amazon ML Challenge 2026 (business entity resolution). Read `CLAUDE.md`, `docs/rules.md`, `docs/strategy.md`, `docs/eda_findings.md` and the `er-playbook` skill (with its references) before you touch code. Everything below must follow `/experiment` (tracked runs, one hypothesis per run, notes, commit).

**Goal:** maximize the *private* leaderboard (unseen test, including France). A number we cannot trust is worth nothing. No leakage, no tuning on test or LB, no cherry-picking.

## Where we are (25 Sep, ~13:00 IST; deadline 27 Sep 23:59 IST)
- Baseline run `20260925-1236_aryan_baseline-v0-keys-lgbm` (`pipelines/baseline_v0.py`):
  - mini F0.5 **0.9057** (US 0.925, India 0.877)
  - pair precision 0.986, pair recall 0.818
  - **blocking recall 0.861** (India 0.827, US 0.883); cands mean 72 / p95 207
- The bottleneck is **blocking recall**. After that, it's recall-at-high-precision in scoring.
- Box: `amlc-box`, r7i.2xlarge (8 vCPU / 61 GB). One heavy job at a time. Run long jobs in tmux with logs in `logs/`. Quota requests for 32 vCPU + GPU are pending.

## Non-negotiable ML hygiene (check each before reporting a number)
1. **Split.** Eval S1 come from fold 0 only (`mini` for iteration, `fold0 minus mini` for confirmation). Models, early stopping, calibration and every learned statistic that uses labels come from folds 1–4 only. Early stopping uses fold 1; training uses folds 2–4.
2. **Pool.** Eval queries always retrieve from the FULL train S2/S3 pool of their country, the same as at test time.
3. **Unsupervised statistics** (token IDF, TF-IDF vocabularies, token rarity) may be fit on the split being processed (train or test). They must never use labels.
4. **Stacking.** Stage-2 features built from stage-1 predictions must use **out-of-fold** stage-1 predictions (K-fold over training S1). For eval and test, use a stage-1 model that never saw those entities.
5. **Record-side competition features** (how many S1s want this record, its rank, the margin) must be computed over ALL S1 of the country. In eval, that means scoring candidates for all train S1 that retrieve the same records (use OOF probs). Computing them inside a chunk only sees a subset: that is a train/test mismatch.
6. **Decision params** (threshold, expected-F calibration, caps) are tuned on `mini` and **confirmed on `fold0 minus mini`**. Report both numbers.
7. **Keeping a change** requires paired bootstrap `p_not_better < 0.05` on identical entities (`scripts/compare_runs.py`), no drop in any country, and no drop in LOCO.
8. **France proxy (LOCO).** For every model/feature change, also train US-only → eval India and India-only → eval US. Features that only help in-country are suspect.
9. **Never** use `country` as a feature. Never add external data. Only MIT/Apache ≤ 8B pretrained weights, with the license verified on the model card and recorded in `docs/decisions.md`.
10. **Test inference** uses exactly the validated code, config and threshold, and writes outputs only through `ber.submission`. `candidate_pairs.tsv` must be the exact set scored.

## Roadmap (phases have gates; don't skip ahead)

### Phase 0: diagnose (≤ 45 min)
- Run `/blocking-audit` and `/error-analysis` on the baseline run.
- Output: which retriever or normalization would recover each class of missed match, and the top FP/FN buckets with their F0.5 cost.
- Build a reusable **candidate cache**: save the eval (mini + fold0) and training candidate sets + features per run, so model/decision experiments don't redo blocking.

### Phase 1: blocking v1 (target: pair recall ≥ 0.97 on mini at ≤ ~100 cands/S1, both countries)
Add retrievers one at a time and log each one's *marginal* recall and pair cost:
1. Char 3-gram TF-IDF (`char_wb`, sublinear, float32) on `n_core`, **per country**, top-k via chunked sparse matmul (or `sparse_dot_topn` if Apache/MIT and it installs). Start with k=30.
2. A second TF-IDF on `n_core + a_street` (name+street). This separates chains and generic names.
3. **Reverse retrieval**: each S2/S3 record → its top-3 S1 by the same TF-IDF. Union these in. This also produces record-side features.
4. Native-script fix: a **learned token map** from training pairs (folds 1–4 only). Align transliterated tokens (e.g. `praivet`→`private`, `limitet`→`limited`, state names in each script) using co-occurrence in matched pairs, then apply it in normalization (bump `NORM_VERSION`, rebuild the cache).
5. Only if recall is still < 0.97: a dense bi-encoder (`intfloat/multilingual-e5-small`, MIT), fine-tuned on folds 1–4 pairs with in-batch + hard negatives, with FAISS top-k. Needs the GPU box or a Kaggle T4.

Other rules for this phase:
- Candidate budget: cap per S1 by a cheap score (max of retriever similarities) so test stays feasible.
- **Estimate test runtime and memory on a 5% test slice before any full test run.**
- Gate: blocking recall up, pairs/S1 reasonable, and a re-trained baseline model on the new candidates improves mini F0.5 (paired bootstrap).

### Phase 2: features v1 + two-stage model (target: better recall at precision ≥ 0.98)
1. Features:
   - TF-IDF cosines (name, name+street)
   - IDF-weighted extra/missing core tokens, with a "business word" flag for extra tokens (hand list: holdings, partners, group, exports, services, enterprises, international, global, industries, solutions, centre/center, associates, trading, ventures…)
   - learned-token-map similarity
   - numeric relations beyond the house number: secondary numbers, unit/PMB, PIN/ZIP
   - alias/domain/handle matches
   - name length ratios
   - per-retriever ranks and the retriever-hit bitmask
2. **Stage 1** LightGBM (fast, all candidates) → OOF probs.
3. **Stage 2** LightGBM on the top candidates, adding context features:
   - rank of the candidate within its S1
   - rank of the S1 within the record
   - margin to the second-best S1 (computed over ALL S1, see hygiene item 5)
   - S2↔S3 agreement: similarity of this candidate to the S1's other high-prob candidates from the other source
   - S1-level stats (max prob, count > 0.5)
4. Try CatBoost/XGBoost only after LightGBM is stable. Ensembling comes later.
5. Gate: mini F0.5 gain confirmed on fold0 minus mini, and LOCO not worse.

### Phase 3: decision layer (cheap, often +0.005–0.02)
- Isotonic calibration on OOF.
- Compare: a global threshold, per-source thresholds, **expected-F0.5 top-k per S1** (`ber.decision.expected_f05_matches`), and a cap of ≤ 10 matches per S1.
- Singleton handling: predict empty when P(no match) is high. Singletons are worth 1.0 each.
- Pick the rule on mini, confirm it on fold0 minus mini.

### Phase 4: France robustness (15% of the test set; no labels)
- French normalization rules (rue/r., bd, av, all., imp, ch, rte, n°, bis/ter; SARL/SAS/SASU/EURL/SCI/SNC/S.A.R.L.; drop "(france)"). Plus a hand-written department↔region map for the cities that appear in test. This is general knowledge, not external data; document it.
- Check LOCO for every Phase 2 feature. Remove features whose LOCO drop exceeds 2× their in-country gain.
- Run test diagnostics for every candidate submission: the France empty rate, mean matches and candidate counts should look like US/India. Big deviations mean a pipeline issue in dense French cities.
- Optional, if time allows: synthetic French pairs made by applying noise operators measured on train pairs to test-France S1 records (inputs only, no labels). Document it clearly.

### Phase 5: neural precision (only after Phases 1–3 are done, and if the GPU quota is approved)
- A cross-encoder (`microsoft/mdeberta-v3-base` MIT or `xlm-roberta-base` MIT) on "name | address" pairs, trained on folds 1–4 hard negatives.
- Score only the uncertain band (0.05 < p < 0.95) and feed its logit into stage 2.
- Run it on the GPU box and stop the box after.

### Phase 6: freeze and ship (27 Sep)
- **18:00 IST feature freeze.** After that, only threshold/ensemble tweaks.
- Final full run (resize to r7i.4xlarge if the quota allows), `/submit`, then `/package-final` by 22:00 IST.

## Submissions (15 total; the human uploads)
- The baseline goes today as the anchor. Afterwards, submit only when a change is **confirmed on fold0 minus mini with a gain ≥ 0.005**.
- Record the val↔LB gap every time (`scripts/record_lb.py`). If the gap drifts, stop and investigate validation before trusting further gains.

## How to work and report
- Start each session with `/status`. Plan the next 2–3 experiments with expected gain and cost.
- **Before any job over 10 min:** estimate its time and memory from a `micro` run, run it in tmux with a log, and report the estimate.
- After each run, report 4 lines:
  1. the change
  2. mini F0.5 (per country) and Δ vs parent with its bootstrap p-value
  3. blocking recall / P / R
  4. keep or kill, and the next step
- Update `docs/ideas_backlog.md`, `docs/decisions.md` and the run notes. Commit and push.
- Use the `validation-auditor` agent before every submission and whenever a gain looks too good (> +0.02 from one change). Use `perf-engineer` when a step won't scale to the 1.73M-S1 test set. Use `research-scout` for technique or license questions.
- Keep credits in mind: stop idle work, and don't leave the GPU box running.
