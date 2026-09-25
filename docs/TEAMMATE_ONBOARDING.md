# Teammate onboarding: from zero to your first experiment in ~45 min

**Where we are:**
- Baseline `20260925-1236_aryan_baseline-v0-keys-lgbm`: val (mini) F0.5 **0.9057**, **public LB 0.901**. The val↔LB gap is small, so our validation is trustworthy.
- Best validated so far: baseline + post-rules = 0.9107.
- In progress on Aryan's box: blocking v1 (recall 0.86 → 0.977 on micro).

Read these first:
- `CLAUDE.md`, for the rules. They are strict: MIT/Apache models only, no external data, no `country` feature.
- `docs/strategy_v2.md`, the problem → solution map.
- `docs/claude_code_master_prompt.md`, the ML hygiene rules.

## 1. Accounts (Aryan does these for you)
- **GitHub:** Aryan adds you as a collaborator on `Aryan-Bhendarkar/Amazon-ML-Challenge-2026`. Accept the invite e-mail.
- **Data:** send Aryan your **12-digit AWS account ID**. He runs:
  `infra\aws\share_bucket.ps1 -Accounts <ids>`

## 2. Your own cloud box (your AWS account and credits; ~15 min)
Follow `infra/aws/README.md`, steps 1, 2, 4 and 5:
1. Upgrade to the Paid plan (the credits carry over).
2. On your laptop:
   - `aws login --region ap-south-1`
   - `$env:AMLC_MEMBER="<yourname>"`
   - `infra\aws\setup_account.ps1 -Email <you>`
   - `infra\aws\devbox.ps1 up`
3. Connect with VS Code Remote-SSH to `amlc-box`, then run:
   ```bash
   AMLC_DATA_BUCKET=amlc26-699191579023 bash ~/box_setup.sh      # GitHub login, venv, data, tests
   ```

## 3. Fast start: pull the prepared caches (skip the ~15 min of rebuilding)
```bash
cd ~/amlc && AMLC_DATA_BUCKET=amlc26-699191579023 bash infra/aws/pull_share.sh
```
You get:
- the raw parquet and folds
- the normalized caches v0 and v1, and the token map
- the **keys_v0 candidate cache**, with features for the train sample, mini and micro
- the **baseline model** (`artifacts/<baseline>/model.lgb`, `features.json`, val predictions)

## 4. Reproduce the baseline (proves your setup works)
```bash
.venv/bin/python pipelines/train_eval.py --cand-ver keys_v0 --subset micro --name smoke-<you>
.venv/bin/python pipelines/train_eval.py --cand-ver keys_v0 --subset mini --name repro-<you> \
    --parent 20260925-1236_aryan_baseline-v0-keys-lgbm --loco
```
Expected: mini ≈ 0.905. Then run `cd ~/amlc && claude` and `/status`.

## 5. Your track (one owner per track, so nobody overwrites anyone)

| Track | Owner | Start from | Goal |
|---|---|---|---|
| A: Blocking v1 → gate run → test submission | Aryan (box 1) | `pipelines/blocking_v1.py` | recall ≥ 0.98 on mini, submission #3 |
| B: Context + difference features | Kaggle teammate | `pipelines/features_v1.py`, `src/ber/ctx_features.py` (strategy_v2 P3–P5) | + gain on keys_v0, then on the v1 cache |
| C: Decision layer (how many matches per S1, singleton head, expected-F) | MLSS teammate A | `pipelines/decision_v1.py`, `src/ber/setdecision.py` (P2) | the oracle says up to +0.024 is available |
| D: France + normalization (French rules, token map v2, honorifics) | MLSS teammate B | `src/ber/normalize.py`, `docs/strategy_v2.md` P6–P7 | LOCO-safe gains, France diagnostics |

Kick off your Claude Code with:
```
Read CLAUDE.md, docs/TEAMMATE_ONBOARDING.md, docs/strategy_v2.md and docs/claude_code_master_prompt.md. I own Track <X>. Plan first (expected gain, time, memory), then run /experiment on the keys_v0 cache. One heavy job at a time.
```

## 6. Team rules (these matter; we're 4 people plus several agents in one repo)
- **Branch per track:** `git checkout -b track-<X>-<you>`. Commit run records together with the code. Push the branch. Aryan merges into `main` after `compare_runs.py` shows a real gain.
- **Only touch files in your track.** Shared library changes (`src/ber/io.py`, `metric.py`, `split.py`, `harness.py`) must be announced first.
- **Nobody uploads to the leaderboard except Aryan** (one login, one device). Send him the run_id + `/evaluate` output.
- **Gains count only if** they're on mini with paired bootstrap p < 0.05, confirmed on fold0 minus mini, and LOCO is not worse.
- **Credits:** stop your box when you're done (`infra\aws\devbox.ps1 stop`). Keep one heavy job at a time.
- **Artifacts** go to your own bucket: `aws s3 sync artifacts/<run_id> s3://$AMLC_BUCKET/artifacts/<run_id>`. Share them with the team via `share_bucket.ps1`.
