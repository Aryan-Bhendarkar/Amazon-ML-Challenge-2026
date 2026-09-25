---
name: status
description: Show Amazon ML Challenge status — hours left to the deadline, leaderboard submissions used today/total, best validation runs, latest runs, and prepared-but-not-uploaded submissions. Use at session start, before planning, and before any submission decision.
---

## Live status
!`python scripts/status.py`

## Leaderboard of runs
!`python scripts/leaderboard.py --top 8`

Using the above, give a 5-line briefing:
1. Time left, and which phase we're in (Day 1 baseline / Day 2 gains / Day 3 freeze after 18:00 IST).
2. Submission budget left today and in total.
3. The current best val run and its gap to the next-best idea.
4. The top 2 items from `docs/ideas_backlog.md` that are not done.
5. One concrete recommendation for the next hour.
