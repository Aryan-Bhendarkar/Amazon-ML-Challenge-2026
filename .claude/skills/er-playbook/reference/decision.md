# Decision rules for macro F0.5

Per S1 with n true matches, predicting k items of which a are correct: `F = 1.25a / (0.25n + k)` (a>0).
- A false positive costs more than a false negative at the same k: for n=3, 3 correct + 1 FP = 0.789, while 2 correct = 0.909.
- Empty on a non-singleton = 0. Any prediction on a singleton = 0.

## Pipeline
1. `assign_best_s1(pred)`: each record keeps only its argmax S1 (+ `margin`).
2. Choose per-S1 outputs:
   - **Global threshold** `t` tuned on `mini` (`tune_threshold`). Robust and simple. Report the curve.
   - **Expected-F0.5 top-k** (`expected_f05_matches`): needs calibrated probs (isotonic). It automatically predicts empty when P(singleton) = ∏(1-p) is high, and adds lower-prob items only when that raises expected F. Usually beats a global threshold when calibration is good.
   - Hybrid: expected-F on calibrated probs, then drop items whose margin to a competing S1 is tiny.
3. Tune on `mini`, confirm on fold0\mini. Never tune on test/LB.

## Useful rules of thumb
- If the S1's best prob < t_low, predict empty. Singletons are 5.6% of train but are pure gold (1.0 each).
- Caps: the train max is 11 matches, so predicting more than ~10 for one S1 is almost certainly a merge error.
- Per-source thresholds (S2 vs S3) are legit: sources have different noise profiles.
- **France hedge**: with no labels, calibration can drift. A slightly higher threshold for France costs little recall and protects precision. Choose it using the LOCO proxy: how much did the optimal threshold shift when evaluated on an unseen country?
