# DIRECTIVES (research lead / Cowork). Re-read this file at the START of every loop iteration.
Updated: 26 Sep 15:05 IST. Aryan has delegated technical decisions to the research lead. Aryan only uploads submissions.

## Standing rules (all lanes)
- **Metrics:**
  - model changes: DM-fold0x Δ ≥ +0.002, clean fold0x Δ ≥ −0.001, LOCO-avg Δ ≥ −0.0005
  - France/transfer changes (Lane C): LOCO-avg Δ ≥ +0.002 AND clean Δ ≥ −0.001
  - one-off micro/mini wins are not enough; confirm on fold0x
- **Workflow:**
  - One heavy job (> 4 threads) per box.
  - Chain jobs in tmux queues so nothing waits for a human.
  - Commit + push after every finished experiment: box2 pushes to a branch `laneC` then opens nothing; box1 merges.
- Log every result in `.claude/experiments.md` (4-line format). Overwrite the top of `.claude/STATUS.md`; keep the Monitor notes.
- **Never:**
  - delete with globs or command substitution
  - touch `student_resource/`
  - open other teams' repos
  - use country as a feature
  - tune on the LB
- **Submission request** = write `SUBMIT-REQUEST:` in STATUS with the path, run_id, clean / DM / LOCO numbers and validator output. The lead approves; Aryan uploads.
- **Stop and write NEEDS-LEAD** in STATUS for any ambiguity. Then continue with the next independent item; don't idle.

## Lane A: amlc-box (box1), the Claude Code session there
1. **NOW (≤15 min): France diagnostic submission (APPROVED by the lead).**
   - Copy the 0953 `matching_results.tsv`, set the match list of every **France** S1 to empty (keep all rows and the tab), write it to `submissions/files/20260926-probe-fr-empty/`, and run the validator `--check-ids`.
   - `candidate_pairs.tsv` is not needed for the LB.
   - Write `SUBMIT-REQUEST: FR-probe` with the formula F_FR = (0.967 − LB_probe)/0.15 + s_FR (s_FR ≈ 0.056 ± 0.01).
2. **B0b (lfrac q05):** finish it and log it. Keep it only if it clears the model gate.
3. **Near-copy discrimination features** (note 11 per-source rank/gap, note 13 per-source thresholds, twin/decoy-pair signal). DM gate.
4. **Lane B, run by this session as a low-CPU background job** (≤2 threads): note 12, frozen bge-m3 band cosines on Kaggle via `/kaggle-gpu`. Hard cutoff 27 Sep 12:00.
5. **21:30 IST: build submission #2 candidate** = best of {B0b, the Lane C transfer-safe feature subset, per-source t} rescored with `--from-feats` (minutes). SUBMIT-REQUEST only if it clears the gate.
6. **23:00 IST: overnight single test featurization** of every kept new feature group (`touch ~/.keepalive`; remove it after).

## Lane C: amlc-box2, the Claude Code session there
1. **LOCO-driven feature-group audit:**
   - Leave one group out, measuring LOCO both directions + clean mini, for groups G1..G5, base families (name sims, addr sims, house, legal, rbits) and lfrac (raw vs q05 vs drop).
   - Output a ranked table and ONE "transfer-safe" feature list (drop/coarsen groups whose in-country gain is < their LOCO loss).
   - It must be rescorable from the saved test_feats (a subset of columns only). Target ready by **20:30 IST**.
2. **French normalization + legal forms** (note 15 list). Deliver it as code on branch `laneC` with unit tests + a label-free FR test-pair change report + LOCO neutrality. It will be applied in Lane A's overnight batch. Target **22:30 IST**.
3. If time remains: test-pool statistics check for France (co-location / name-count feature distributions FR vs US/IN on test).

## Timeline (IST)
| When | Milestone |
|---|---|
| 26 Sep 15:30 | FR probe upload (submission 3/5 today) |
| 26 Sep 20:30 | LOCO audit done |
| 26 Sep 21:30–22:30 | submission #2 candidate (4/5 today) |
| 26 Sep 23:00 → 04:00 | overnight test featurization |
| 27 Sep 09:00–12:00 | merge + seed-bag; bge-m3 cutoff 12:00 |
| 27 Sep 14:00 | final candidate |
| 27 Sep 18:00 | FREEZE |
| 27 Sep 20:00 | final rescoring + validator |
| 27 Sep 22:00 | final zip |
