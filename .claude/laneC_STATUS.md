Updated: 26 Sep 15:55 IST (amlc-box2, Lane C; branch `laneC`)

Lane-specific file on purpose: `.claude/STATUS.md` / `.claude/experiments.md` do not exist in origin, so writing
them on `laneC` would collide with box1's local copies at merge time. Box1: fold these into the shared files.

Current: LOCO audit round 1 running (tmux `audit`, 14 drop-one-group variants @ seed 42 + full @ 42/43/44), round 2
queued (tmux `audit2`: -B_name, -B_addr, -B_retr, -B_fmt, q05, -G3_lfrac_raw, -G3_ldf). ETA r1 ≈ 17:30, r2 ≈ 18:30 IST.
Then: clean mini + LOCO confirmation of the transfer-safe list via `features_v1.py --drop-feats/--lfrac-q` (≈ 19:15).

Done:
- LOCO reference reproduced on box2 exactly: India→US 0.95425, US→India 0.92988 (avg 0.94207). Seed noise of the
  LOCO-avg (seeds 42/43/44): tuned 0.94207/0.94243/0.94277 (spread 0.0007), at source t 0.94063/0.93944/0.93974
  (spread 0.0012). The +0.002 Lane C gate is > 2× the seed spread.
- **norm v2 (French normalization + legal forms)**: code + tests on `laneC`; US/India byte-identical (0 changed train
  rows, exact) → LOCO Δ = 0, clean Δ = 0 by construction; label-free FR report `docs/france_norm_v2.md`
  (S1 with a strong candidate 0.908 → 0.976; US 0.961, India 0.968). Apply path for Lane A's overnight batch:
  `build_norm_cache.py --split test` → `refeat_norm.py --tag test` → `predict_test_v1 --src-tag test_n2 --norm 2`.
  `refeat_norm` verified: full-mini recompute IDENTICAL to the cache (8.38M pairs).
- Test-pool statistics (priority 3): France is 3× more co-located (S1 address shared 0.19 vs 0.05–0.07) and has
  3–6× denser generic name tokens (median max token df/n 0.027 vs 0.005–0.010) than US/India, which look alike →
  LOCO cannot see the G2/G3-lfrac shifts. Table in `docs/france_norm_v2.md`.

NEEDS-LEAD:
1. Monitor notes 14 and 15 are not on box2 / origin (latest pushed note is 8 in `claude/STATUS.md`; commits mention
   10). I worked from `.claude/DIRECTIVES.md` (Lane C items) + HANDOFF EXP-F. If note 15's French list has items
   beyond docs/france_norm_v2.md, push it and I will add them.
2. norm v2 cannot pass the literal Lane C gate "LOCO-avg Δ ≥ +0.002": it is LOCO-neutral by construction (it does not
   touch US/India). Evidence is the label-free FR report. Proposed: accept on "LOCO = 0 exactly + FR report".

Submission requests: none.
