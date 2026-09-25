# Teammate onboarding: from zero to your first experiment in ~45 min

**Where we are:**
- Baseline `20260925-1236_aryan_baseline-v0-keys-lgbm`: val (mini) F0.5 **0.9057**, **public LB 0.901**. The val↔LB gap is small, so our validation is trustworthy.
- Best validated so far: baseline + post-rules = 0.9107.
- In progress on Aryan's box: blocking v1 (recall 0.86 → 0.977 on micro).

Read these first:
- `CLAUDE.md`, for the rules. They are strict: MIT/Apache models only, no external data, no `country` feature.
- `docs/strategy_v2.md`, the problem → solution map.
- `docs/claude_code_master_prompt.md`, the ML hygiene rules.

## 1. Access model: read-only, nothing touches Aryan's repo
- You get a **read-only snapshot** of the code and a **read-only** data pack from Aryan's S3 bucket. The bucket policy only allows `GetObject`/`ListBucket` for your AWS account.
- You work in **your own local git** (optionally your own private GitHub repo). You **never push to Aryan's repo.**
- Your results come back as a **patch + run record** (step 7). Aryan's Claude Code reviews them and merges what is proven.
- To get access, send Aryan your **12-digit AWS account ID**. He runs `infra\aws\share_bucket.ps1 -Accounts <ids>`.

## 2. Get the code snapshot (on your laptop, after `aws login --region ap-south-1`)
```powershell
mkdir amlc; cd amlc
aws s3 cp s3://amlc26-699191579023/share/code/amlc-latest.tar.gz . ; tar -xzf amlc-latest.tar.gz ; del amlc-latest.tar.gz
git init -q; git add -A; git commit -qm "snapshot from Aryan"
```

## 3. Your own cloud box (your AWS account and credits; ~15 min)
1. Upgrade to the Paid plan (credits carry over).
2. From the snapshot folder on your laptop:
   ```powershell
   $env:AMLC_MEMBER="<yourname>"
   infra\aws\setup_account.ps1 -Email <you>
   infra\aws\devbox.ps1 up
   ```
3. VS Code Remote-SSH → `amlc-box`. In the box terminal:
   ```bash
   mkdir -p ~/amlc && cd ~/amlc
   aws s3 cp s3://amlc26-699191579023/share/code/amlc-latest.tar.gz - | tar -xz
   git init -q && git add -A && git commit -qm "snapshot from Aryan"
   AMLC_DATA_BUCKET=amlc26-699191579023 bash ~/box_setup.sh      # venv, data, caches, tests (skips the GitHub clone)
   AMLC_DATA_BUCKET=amlc26-699191579023 bash infra/aws/pull_share.sh   # prepared caches + baseline model
   ```

The fast-start pack gives you:
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

## 6. Team rules
- **Only work on your track's files.** Keep changes small and focused, so Aryan can merge them cleanly.
- **Nobody uploads to the leaderboard except Aryan** (one login, one device).
- **Gains count only if** they're on mini with paired bootstrap p < 0.05, confirmed on fold0 minus mini, and LOCO is not worse.
- **Credits:** stop your box when done (`infra\aws\devbox.ps1 stop`). One heavy job at a time.

## 7. Handing results back (how your work reaches the main pipeline)
When a run beats its parent:
```bash
cd ~/amlc
git add -A && git commit -m "<run_id>: <what> (mini F0.5 x.xxxx)"
git format-patch -1 -o ~/handoff/                                   # your code change as a patch file
tar -czf ~/handoff/<run_id>.tar.gz experiments/runs/<run_id>         # the run record (meta/metrics/notes)
aws s3 cp --recursive ~/handoff s3://$AMLC_BUCKET/handoff/<you>/     # your own bucket
```
Then send Aryan the run_id and your bucket name, and run `share_bucket.ps1 -Accounts 699191579023` in your account so he can read it.
He reviews it with the validation-auditor, re-runs it, and merges it. New snapshots appear at `share/code/amlc-latest.tar.gz`. To update, re-download it into a fresh folder, or apply it over yours.
