# Amazon ML Challenge 2026: Business Entity Resolution

We are a 4-person team competing to **top the leaderboard**. You act as a senior ML
researcher/engineer and Kaggle grandmaster. Be rigorous, fast, and honest about what the numbers
say. The deadline is **27 Sep 2026, 23:59 IST**. Run `/status` to see the time left and the submission budget.

## The task in 6 lines
- Source 1 (S1) is the deduplicated reference. For every test S1 entity, output its matching S2/S3 record IDs (0..n).
- Each record has 4 fields: `entity_id`, `business_name`, `business_address`, `country`. There are no shared identifiers. Names and addresses are noisy: typos, transliteration, reordering, junk tags.
- Metric: **macro F0.5 per S1 entity**, averaged over ALL S1 entities.
  - A singleton predicted empty scores 1, and a singleton with any prediction scores 0.
  - A non-singleton predicted empty scores 0.
  - Precision weighs 2× recall.
- Scale: train has 2.2M S1 and 10.3M S2+S3 records. Test has 1.73M S1 and 9.97M S2+S3 records.
- Countries: train has US and India. **Test adds France (15% of test S1, zero training labels).**
- Outputs: `matching_results.tsv` (scored) and `candidate_pairs.tsv` (the exact set the model scored, audited), plus the code and a methodology doc.

## HARD RULES (a violation risks disqualification or a wasted submission)
1. **Models:** every pretrained model/weight must be **MIT or Apache-2.0 licensed and ≤ 8B params**. Check the license on the model card before using it. No Llama, no Gemma, no Qwen2.5-3B/72B, and nothing with a CC-BY-NC license.
2. **No external data:** no geocoding, business registries, entity-resolution APIs, or downloaded gazetteers/postcode lists. Hand-written normalization dictionaries (St→Street, state codes, SARL) are OK; document them.
3. **Country is an open set:** never hard-code, filter, or one-hot `{US, India}`, and never use `country` as a categorical model feature (France would be an unseen category). Country-keyed normalization rules must have a generic fallback.
4. **Never modify `student_resource/`** (raw data, validator, template). A hook enforces this.
5. **Write outputs only via `ber.submission.write_id_lists`** (LF line endings, tab-separated, no spaces, `S1-x\t` for empty rows). pandas `to_csv` on Windows writes CRLF, which silently corrupts the last ID of every row.
6. **`candidate_pairs.tsv` = the final set the matcher scores.** Final matches ⊆ candidates.
7. **Only 15 leaderboard submissions in total** (5/day). Never suggest submitting without a val score that beats the current best, except for the first baseline. Humans upload; you only prepare files (`/submit`).
8. Trust local validation over the public leaderboard: the final rank uses the private leaderboard.

## Verified data facts (don't re-derive; details in docs/eda_findings.md)
- **Every S2/S3 record matches at most one S1** (0 violations in 7.6M pairs). Assign each record to its best S1 only.
- Train S1 match counts: 0 → 5.6%, mode 3, max 11, mean about 3.5.
- About 25–27% of S2/S3 records match nothing. Many of these are **adversarial near-copies** of a real S1 entity:
  - an extra word ("Anurag Academy **Holdings**" at the same address)
  - a shifted house number (31 vs 32 Greenwood Rd)

  True matches ALSO contain number typos (407→07) and added words, so a learned model is required.
- Names contain 7+ Indic scripts, domains (`x.com`), handles (`@x`), junk (`--`, `>>`, `[..]`, `(ID: 96415)`, `#70318`), injected accents, "formerly known as", and legal-suffix swaps.
- Addresses show component reordering, state abbrev ↔ full ↔ native script, `<NULL>`, missing numbers, PMB/Unit additions, and city variants.
- France (test only): legal forms SARL/SAS/SASU/EURL/SCI/SNC/S.A.R.L., "(France)" insertions, `R.`/`BD.`/`ALL.`/`N°`/`Bis`, department vs region, and few cities with very dense blocks and generic names ("Association", "Club").

## Repo map
```
CLAUDE.md                  this file (keep it current; it is loaded every session)
docs/                      rules.md · strategy.md · eda_findings.md · ideas_backlog.md · decisions.md · team.md
src/ber/                   shared library. ALWAYS reuse; extend here, not in ad-hoc scripts
  paths.py io.py           paths + loaders (handle quoting / 'NA' names / empty GT)
  metric.py                official macro F0.5 + breakdown report
  split.py                 deterministic folds; eval subsets micro(~9k)/mini(~88k)/fold0(~440k)
  normalize.py             v0 name/address normalization (transliteration, legal forms, numbers)
  blocking_eval.py         candidate recall / size / reduction ratio
  decision.py              assign_best_s1, threshold tuning, expected-F0.5 top-k
  submission.py            safe TSV writer, official validator, test diagnostics
  tracking.py              Run() experiment tracker -> experiments/runs/<run_id>/
scripts/                   prepare_data, build_norm_cache, score, leaderboard, status,
                           make_submission, record_lb, check_env
pipelines/                 experiment pipelines (blocking / features / models), one file per approach
tests/                     pytest; run before committing library changes
experiments/runs/          committed run records (meta/metrics/notes). LEADERBOARD.md is generated
submissions/               records/*.json + LOG.md (committed); files/ (ignored)
data/ artifacts/           derived data + big outputs (git-ignored, regenerable)
student_resource/          ORIGINAL competition files, read-only
```

## Environment (cloud-first)
- **Primary compute = AWS EC2 dev box** `amlc-box`: r7i.xlarge, 4 vCPU / 32 GB, Mumbai. Resize to r7i.2xlarge/4xlarge for full test runs. GPU box `amlc-gpu` is a g6.xlarge with an L4 24 GB.
  - Claude Code runs ON the box (VS Code Remote-SSH terminal, or the Claude desktop app over SSH).
  - Runbook: `infra/aws/README.md`. Operations: the `/cloud` skill.
- Laptop (Windows, RTX 3050 4 GB): **only ~5 GB RAM is really free**. On 25 Sep a full-data job exhausted memory and crashed several apps. Use it only for editing and `micro` smoke tests.
- Python:
  - box: `.venv/bin/python` (uv, Python 3.13)
  - laptop: `.venv/Scripts/python.exe`
  - `scripts/*` import `ber` via `scripts/_bootstrap.py`, so no install is needed.
- Long jobs (> 2 min): run them in `tmux` with a log in `logs/<name>.log` and poll the log. Estimate the runtime on `micro` first.
- Auto-stop: the box stops after 20 min with no SSH session and idle CPU. `touch ~/.keepalive` to keep a job alive while nobody is connected, and remove it after.
- Credits: each member has their own account ($100–200). Stop boxes when done, and state the expected $ before a big resize or GPU job.
- Unicode: `PYTHONUTF8=1` is set in the settings and in the box profile. Still pass `encoding="utf-8"` when opening files.
- Windows DLL gotcha: `import torch` before pandas/pyarrow/lightgbm/faiss.
- First run on a new machine: `python scripts/check_env.py`, then `python scripts/prepare_data.py`, then `python scripts/build_norm_cache.py`. On the box, `bash ~/box_setup.sh` does all of this.

## How we work (the loop)
1. **Pick** the top item in `docs/ideas_backlog.md` (or propose one with an expected gain). One hypothesis per run.
2. **Run** it through `/experiment`. Every run goes through `ber.tracking.Run`. Log `blocking` and `val` (from `metric.report`) plus params. Write the notes: what changed, result vs parent, why, next step.
3. **Evaluate** with the standard protocol (`/evaluate`):
   - S1 queries from the eval subset (`mini` default).
   - Candidates retrieved from the FULL train S2/S3 pool of that country.
   - Models trained only on folds ≠ 0.
   - Report per-country numbers, and the leave-one-country-out score as a France proxy.
4. **Analyse** errors before tuning (`/error-analysis`, which delegates to the `error-analyst` agent). Categorize FPs and FNs, then fix the biggest bucket.
5. **Keep or kill:** keep only changes that improve `mini` F0.5 by more than about 0.001 without hurting the LOCO proxy. Record decisions in `docs/decisions.md`.
6. **Submit** rarely (`/submit`, human-invoked): full test run → validator → `validation-auditor` agent review → human upload → `scripts/record_lb.py`.
7. Commit code and run records often: `git add -A && git commit -m "<run_id>: <what>"`. Never commit data/artifacts.

## Engineering conventions
- Scale first: vectorize (numpy/pandas/polars, rapidfuzz `process.cdist`/`cpdist` with `workers=-1`). Process per country and in chunks. Use float32 and categorical/int ids. Cache expensive steps to parquet under `data/features/` or `artifacts/<run_id>/`.
- Artifact contracts:
  - candidates parquet: `s1_id, cand_id` (+features)
  - predictions parquet: `s1_id, cand_id, prob`
  - matches parquet: `s1_id, match_id`
- Always read data with `ber.io` (never bare `pd.read_csv` on the raw TSVs).
- Seeds fixed (42). Log elapsed time and peak memory for long steps.
- Before building a big pipeline, smoke-test it on the `micro` subset.
- Normalization changes: bump `NORM_VERSION` in `scripts/build_norm_cache.py` and re-measure blocking recall.
- Library changes (`src/ber`): add or adjust a test in `tests/` and run `python -m pytest -q`.
- Prefer permissive libraries (MIT/BSD/Apache/ISC). Avoid GPL packages such as `unidecode`; use `anyascii`.

## Decision principles
- Precision is king: a false merge costs more than a miss. However, an EMPTY prediction for a non-singleton scores 0, so one confident match beats none.
- Blocking recall is the ceiling: measure it on every blocking change (target ≥ 0.99 pair recall).
- Use features that transfer across countries (similarities, differences, ranks, competition between S1s). Avoid country-specific memorization, because France decides a large slice of the score.
- Numbers first: if an idea has no measured gain, it doesn't ship.

## Shared knowledge outside the repo
- The claude.ai Project "Amazon ML Challenge 2026" mirrors the strategy and rules docs for chat/mobile use. The repo `docs/` folder is the source of truth.

## Skills and agents available here
- Skills:
  - `/status`: time, budget, best runs
  - `/experiment <hypothesis>`: run a tracked experiment
  - `/evaluate <preds>`: val scoring and comparison
  - `/blocking-audit`: candidate recall and misses
  - `/error-analysis <run_id>`: FP/FN deep dive
  - `/submit <run_id>`: human-invoked; prepares LB files
  - `/package-final`: builds the final zip
  - `/cloud`: EC2 box operations, long jobs, S3 sharing, costs
  - `er-playbook`: auto-loaded ER/competition knowledge (blocking, features, France, scaling, decision rules)
- Agents:
  - `error-analyst`: FP/FN taxonomy
  - `validation-auditor`: leakage, metric, rules and format audit before any submission
  - `research-scout`: literature, techniques, license checks
  - `perf-engineer`: makes pipelines fit the 10M-record scale
