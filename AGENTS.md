# AGENTS.md: Codex experimenter (Amazon ML Challenge 2026, Business Entity Resolution)

## Your mission
Beat our best public LB score, **0.9795**, with a genuinely better model or pipeline change.
- You have full freedom of approach.
- Work on branch `codex` and push only to `origin codex`.
- The LB closes 27 Sep 23:59 IST. Code freeze is 18:00 IST, so a result must be validated and test-scored before then to count.

## Where to run things (important)
- **This laptop has only ~2 GB of free RAM.** Full-data jobs crashed it on 25 Sep. Use it for editing code and git only.
- **Heavy compute:** `ssh amlc-box2` (AWS, 8 vCPU / 61 GB).
  - `~/amlc` on box2 is the team's working copy (branch `audit`). Do not edit it.
  - Clone your branch next to it: `git clone -b codex https://github.com/Aryan-Bhendarkar/Amazon-ML-Challenge-2026.git ~/amlc_cx`.
  - Then symlink the shared data: `ln -s ~/amlc/data ~/amlc_cx/data; ln -s ~/amlc/artifacts ~/amlc_cx/artifacts; ln -s ~/amlc/.venv ~/amlc_cx/.venv; ln -s ~/amlc/student_resource ~/amlc_cx/student_resource`.
  - Shared files: READ anything; WRITE only NEW files whose name starts with `cx_`; never overwrite or delete existing files.
  - Run jobs in tmux sessions named `cx_*`.
- **Never touch `amlc-box`** (box1). It builds the team's submission files.
- **GPU:** Aryan will give you a separate Kaggle account if needed. Use the `scripts/kaggle_gpu.py` workflow with a private dataset and kernel.

## Current best (read these first)
- `.claude/AUDIT_REPORT.md`: the full error analysis of the best model D and the dead ends.
  - 86% of the loss is recall.
  - In-candidate rejects are 45% of the loss, mostly empty-address records.
  - Blocking misses are 29%, e.g. trade name ≠ legal name at the same address; India has 2× more than US.
  - The decision-layer rules are all dead.
- `docs/METHODOLOGY.md` and `docs/REPRODUCE.md`: the final pipeline end to end.
- `CLAUDE.md`: the rules. The tails of `.claude/experiments.md` and `.claude/STATUS.md`: what was tried and killed, with numbers.
- **Pipeline:** blocking v1_n2 (recall 0.987) → ~100 GBDT features + cross-encoder logits (MiniLM / XLM-R fine-tuned on Kaggle) → LightGBM → t = 0.825 → one-owner assignment. Countries unseen in training (France) fall back to the GBDT without cross-encoder features.
- **Key LB fact:** the cross-encoder was worth +0.0135 on US/IN test but only +0.004 on validation. Test has ~19% of S1 removed while their records were kept (orphans), and val under-represents this. Signals that help reject orphans pay off on the LB 3× more than val shows.
- **Validation:** DM-fold0x via `pipelines/dm_val.py`. The current best, D, scores 0.98485. Keep a change only if the paired-bootstrap CI > 0 and no country drops.

## Hard rules (disqualification risk)
- Pretrained models: MIT or Apache-2.0 only, ≤ 8B params.
- No external data or lookups: no geocoding, registries, or downloaded gazetteers.
- `country` is never a model feature; never hard-code US/India/France.
- No leakage: labels come from train folds only; stacking is out-of-fold.
- Never modify `student_resource/`.
- `candidate_pairs.tsv` must be exactly the scored set, and matches ⊆ candidates.
- Kaggle datasets and kernels stay PRIVATE.
- Never upload to the leaderboard; Aryan does that.

## Deliverable
Push `CODEX_REPORT.md` on branch `codex` with:
- what you changed
- DM-fold0x delta with CI and per country
- if it wins: the exact commands to build the test `matching_results.tsv` + `candidate_pairs.tsv`, and the validator output
