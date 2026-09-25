# Team workflow

## Roles (adjust names)
| who | owns |
|---|---|
| Aryan | pipeline integration, validation protocol, AWS/SageMaker, submissions (the ONLY uploader, one laptop) |
| Kaggle teammate | features + LightGBM, decision rules |
| MLSS teammate A | bi-encoder retrieval (EXP-012), learned token map |
| MLSS teammate B | cross-encoder (EXP-015), France strategy (EXP-020–022) |
| Claude | first versions, error analysis, reviews, strategy calls |

## Git
- One private GitHub repo. `main` is always runnable. Short-lived branches per experiment: `exp/<author>-<slug>`.
- Commit run records (`experiments/runs/*`) with the code that produced them. Commit message: `<run_id>: <what changed> (mini F0.5 x.xxxx)`.
- Never commit `data/`, `artifacts/`, `student_resource/dataset/`, `.venv/`, or submission TSVs.
- Pull before starting a run. Run ids include the author, so records never conflict.

## Each teammate's machine
1. Clone the repo and copy the competition `student_resource/` folder into the repo root (or set `AMLC_RAW_DIR`).
2. Create the environment:
   - `python -m venv .venv --system-site-packages`
   - `.venv/Scripts/python -m pip install -r requirements.txt` (Linux: `.venv/bin/python`)
3. Prepare data:
   - `python scripts/check_env.py`
   - `python scripts/prepare_data.py`
   - `python scripts/build_norm_cache.py`
4. Start Claude Code in the repo root and run `/status`.
5. Set your name once: `git config user.name "<name>"` (used in run ids), or set `AMLC_AUTHOR`.

## Sync rhythm
- Every ~3 h: 10-minute sync. Review the leaderboard (`python scripts/leaderboard.py`), reprioritize `docs/ideas_backlog.md`, and decide on the next submission.
