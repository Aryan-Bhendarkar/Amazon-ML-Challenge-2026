# Final workflow: 0.961 (val) → 0.98+ (25 Sep 21:00 IST, ~51 h left)

## Where we are (verified)
| run | mini F0.5 | India | US | P | R | blocking recall |
|---|---|---|---|---|---|---|
| baseline (LB 0.901) | 0.9057 | 0.877 | 0.925 | 0.986 | 0.818 | 0.861 |
| + post-rules (LB 0.9046) | 0.9107 | | | 0.989 | 0.825 | 0.861 |
| G1–G5 features on OLD cands | 0.9209 | 0.895 | 0.938 | 0.996 | 0.840 | 0.861 |
| **blocking v1 + baseline feats (gate)** | **0.9611** | 0.953 | 0.967 | 0.987 | 0.916 | **0.984** |

- The val→LB gap is about −0.005, so validation is trustworthy.
- The test run for the gate model is running (tmux `v1test`, about 2.5 h, `--save-feats`). It becomes **sub #3**, expected LB about 0.955.
- Remaining loss is about 0.039:
  - India recall (blocking 97.5%)
  - precision dropped to 0.987 with more candidates; the G1–G5 features fix exactly this (0.9956 on the old cands)
  - the per-S1 cutoff (the oracle said +0.024 on the baseline)

## What we take from the external research, and what we reject
**Adopt (it applies to our data):**
1. Isotonic calibration on OOF, then **per-S1 expected-F0.5** selection. For *macro* per-entity F this is the right generalization of the `T* = 0.8·F*` rule; our tuned t = 0.70–0.725 already sits near 0.8 × 0.96.
2. **Monotone constraints** on similarity features in LightGBM. This is a precision and robustness guardrail, and matters most for France (unseen).
3. More name features:
   - **Soft-TFIDF / Monge-Elkan** (IDF-weighted JW token alignment)
   - a **rare-token disagreement** flag
   - prefix/superstring "division" pattern
4. **Hub suppression.** This is our `s1_same_name` / `s1_same_addr` counts. Also add a stricter margin for S1s with a high share of generic names.
5. **Mutual-best / margin.** Record-side competition over ALL S1: the best-other-S1 score and the margin. A "close call → abstain" rule is right for F0.5.
6. **Triangle consistency, done correctly for our star structure.** A candidate supported by a confident *sibling from the other source* (S2↔S3 similarity among an S1's candidates) gets a boost feature. The pairwise classifier can't see this.
7. **Cross-encoder with hard negatives**, field-structured input, fed as one calibrated scalar into the GBDT. Only on the uncertain band, and only if a GPU is available.
8. Group-level CV. We already do this: folds by S1, so no pair-level leakage.

**Reject (doesn't apply here, or costs more than it returns):**
- Tax ID / VAT / phone / DUNS vetoes: those fields don't exist in this data (only name, address, country).
- HAC / Leiden / correlation clustering: our structure is a **star** (a deduplicated S1 reference, and every record has at most one owner). The correct formulation is *assignment* (`assign_best_s1`), not clustering. Clustering can only add chaining risk.
- "Most entities are singletons": false here (5.6%). The singleton head is still useful.
- Focal loss / `scale_pos_weight < 1`: equivalent to moving the threshold, and it breaks calibration, which expected-F needs.
- Node2Vec / GraphSAGE embeddings, SPLADE, weighted MinHash: high cost and leakage risk. TF-IDF top-k already reaches 98.4%.

## The loop that makes iteration fast
- **Test candidates + features are cached once** (`data/cands/v1_n1/test.parquet` from `--save-feats`). After that, every new model or decision rule is **train on the mini cache → rescore the cached test features → make_submission** in about 15–25 min, not 2.5 h.
- New features must be computable on the cached pairs, so blocking stays frozen at v1_n1. Blocking changes need a new test blocking pass; that goes into **one** batch only (Stage 4).
- One heavy job at a time on amlc-box. The standard vCPU quota request (32) is pending. If it's approved, run heavy training on a second box.

## Stages (each gate = a mini gain with bootstrap p < 0.05, confirmed on fold0x, and LOCO not worse)
**Stage 1 (tonight): features on the new candidates (expected +0.010–0.015)**
- `features_v1 --cache v1_n1 --groups G1..G5`, with the LOCO crash fixed.
- Then compute the same features for the cached test pairs.

**Stage 2 (tonight/early morning): decision layer (expected +0.005–0.012)**
- OOF stage-1 (5-fold over folds 1–4 training S1).
- Isotonic calibration.
- Stage-2 LightGBM with relative/context features: rank in S1, rank in source, gap to best, record-side best-other-S1 margin over all S1, S2↔S3 sibling support.
- Per-S1 expected-F0.5 vs global t vs k-model. Pick on mini, confirm on fold0x.
- Fold the post-rules into features; keep them as rules only if they still add.

**Stage 3 (Sat morning): precision polish (expected +0.003–0.008)**
- Soft-TFIDF / Monge-Elkan, rare-token disagreement, division-prefix pattern.
- Monotone constraints.
- A second GBDT (CatBoost), rank-averaged.
- Margin abstain rule.

**Stage 4 (Sat daytime): India recall (expected +0.004–0.008)**
- Analyse the 2.5% India blocking misses: `/blocking-audit` on v1_n1.
- Add targeted retrievers (reverse retrieval on India only; a fine-tuned multilingual-e5-small dense catch-net on native-script and hard records).
- One new test blocking pass (about 2.5 h) → cache → rescore.

**Stage 5 (Sat evening, optional): cross-encoder on the uncertain band** (GPU box or Kaggle T4; mDeBERTa-v3-base, MIT).

**Stage 6 (Sun):** France check (French normalization, per-country test diagnostics), final ensemble, **freeze 18:00 IST**, `/package-final` by 22:00 IST.

## Submission ladder (about 13 left)
1. sub #3 = gate (tonight)
2. Stage 1
3. Stage 1 + 2
4. + Stage 3
5. + Stage 4
6. final

Submit only when fold0x confirms the gain. Record the LB after each submission.
