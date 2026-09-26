# COWORK BRIEF: AMLC 2026, research lead and monitor (26–27 Sep 2026)

You are the **research lead and monitor** for our Amazon ML Challenge 2026 team. You work in the same project folder as Claude Code. Claude Code does the implementation and follows `claude/HANDOFF.md`. Your job is to **think, check, decide, and hand off**. You do not write the pipeline yourself.

Start by reading, in this order:
1. `claude/rules_and_gotchas.md`
2. `claude/strategy.md`
3. `claude/research_brief.md`
4. `claude/HANDOFF.md`
5. `claude/STATUS.md` and `claude/experiments.md`, if they exist

Then follow §5 ("How to work").

---

## 1. Where we are
- Public LB **0.967**; val 0.9813 (P 0.996, R ~0.952); top of the LB ~0.987.
- Deadline **27 Sep 23:59 IST**; feature freeze 18:00; final zip by 22:00.
- Submissions left: **2 today, 5 tomorrow**. We submit only after a validated improvement.
- **Metric:** macro F0.5 per S1. A false positive on a true singleton costs that entity's full score.
- **Structure:** every S2/S3 record belongs to at most one S1.
- **Main suspect: the orphan shift.**
  - Test has ~5.76 records per S1, train 4.67.
  - 4.67 / 0.81 = 5.77, which fits ~19% of S1 being removed while their records were kept.
  - Those records then get matched to similar surviving S1s (chains, co-located or templated businesses).

## 2. Research synthesis: three AI reports compared
We asked Claude, ChatGPT and Gemini the same research brief. Here is what we decided from them. Treat it as settled unless new measurements say otherwise.

**All credible sources agree on:**
- **Fix orphans first.** Train and validate on S1 pools with some owners removed. The evidence:
  - entity-linking "NIL" work: *Learn to Not Link* (ACL Findings 2023); *BLINKout* (CIKM 2023, with KB pruning)
  - partial-KB inference (arXiv 2303.10330), where precision collapses when the KB is incomplete
- **No global assignment (Hungarian or min-cost flow).** S1 can own any number of records, so the problem reduces to: each record goes to its best S1, or to none.
- **No threshold re-estimation from test scores** (EM/SLD, BBSE, GMM). The orphans are a new kind of negative that the model scores confidently, so these methods can't see them.
- **Cross-encoder: selective use only.** Ours was +0.004 in-country but −0.002 in the leave-one-country-out (LOCO) test. The likely cause (arXiv 2607.24688) is shortcut learning on how often exact name/address agreement appears in negatives. France has ~19% shared addresses versus 6–7% in train.
- **France alias mining from test: one pass, human-reviewed.** Never iterate it.

**Where they differed, and the verdicts:**

| Idea | Source | Verdict |
|---|---|---|
| Retrain one model on pruned pools, recomputing competition features | Claude | **Adopt, #1** |
| Separate "owner exists" gate multiplied with the pair probability | ChatGPT | Only a fallback. It double-counts the same evidence and breaks calibration. |
| Record-record sub-cluster features (orphan cluster detected without its owner) | Claude | **Adopt, #2.** The only orphan signal that doesn't need the missing owner. |
| Top1/top2 margins, record exclusivity | ChatGPT | Already built. The new value is recomputing them under pruning (part of #1). |
| 2-hop sibling rescue + reverse record→S1 retrieval | Claude | **Adopt.** In-candidate false negatives are the biggest recall bucket. |
| Multilingual embedding blocker (potion-multilingual-128M) | ChatGPT | Adopt only if it recovers ≥30% of missed pairs on mini. Verify the license first. |
| Fellegi–Sunter LLR features | ChatGPT | Skip; LightGBM already learns bin log-odds. |
| NFD ligature fix (œ→oe, æ→ae); `Ste`=sainte in addresses vs `Sté`=société in names | Gemini (corrected) | **Adopt** (cheap). |
| Poisson-binomial/GMM/Youden thresholds, dual-head orphan cross-encoder on the S1 side, test-time BatchNorm, Noisy Student, mDeBERTa, coordinate features | Gemini | **Reject.** Gemini put the orphans on the wrong side, used the wrong metric, relied on features we don't have, and ignored the deadline. |

**Warning:** ChatGPT cited a public "amazon-ml-challenge" GitHub repo. Nobody opens other teams' code; our code is audited.

**Realistic target:** +0.006 to +0.012 (≈0.973–0.979). Reaching 0.987 would also need France to land cleanly.

## 3. The plan Claude Code is executing (details in HANDOFF.md)
- **EXP-A:** orphan-simulated validation. Remove 19% of S1 from the whole competing pool, keep their records, and recompute competition features. Run the current model on it to test the hypothesis.
- **EXP-B:** retrain on pools with {0, 19, 30}% of S1 removed (half at random, half biased toward chains and co-located S1). Candidate for **submission #1**.
- **EXP-C:** sub-cluster features.
- **EXP-D:** threshold + singleton guard picked on orphan-simulated out-of-fold scores. B+C+D is the candidate for **submission #2**.
- **EXP-E (27 Sep):** recall: 2-hop rescue, reverse retrieval, and optionally the embedding blocker.
- **EXP-F:** France: test-pool IDF and density statistics, normalization fixes, alias proposals for human review, and the name-twin feature.
- **EXP-G (optional):** cross-encoder as a narrow add-rule, killed if LOCO < 0.
- **EXP-H:** seed bagging, monotone constraints, adversarial check.
- **Primary metric:** orphan-simulated fold0x. Guardrails: clean fold0x and LOCO.

## 4. Your decision rules
- **After EXP-A:**
  - Orphan drop ≥ 0.006: confirmed, continue B→D.
  - Drop < 0.004: orphans are not the main gap. Tell Claude Code to go to E and F and re-plan with the human.
  - Between 0.004 and 0.006: continue with B, but also start E in parallel.
- **Keep/kill:** use the paired bootstrap on identical S1s.
  - Keep a change if orphan Δ ≥ its gate (see HANDOFF), clean Δ ≥ −0.001 and LOCO Δ ≥ −0.0005.
  - A change that helps in-country but hurts LOCO is killed (France is 15% of test and has no labels).
- **Submission gates:**
  - #1: orphan fold0x ≥ +0.003 vs the current LB model.
  - #2: a further ≥ +0.002 on top of #1.
  - Otherwise wait for tomorrow. Never pick a model by public LB.
- **The final pick** at 18:00 on 27 Sep is the best orphan-simulated fold0x model that is neutral on the guardrails.

## 5. How to work
1. **Check Claude Code's progress** after each step: read `claude/STATUS.md` and `claude/experiments.md`.
2. **Sanity-check every result before accepting it.** Look for:
   - **Leakage:** labels from eval folds used in features or thresholds; test labels can't exist.
   - **Orphan-sim correctness:** the removal is applied to the competing pool, not just the queries; features are recomputed after removal; the removed owners' records are labelled unmatched; the simulated records-per-S1 ratio is close to 5.7.
   - **Suspicious gains:** > +0.01 from one change usually means a bug.
   - **Output format:**
     - `\n` line endings
     - an empty row still has its tab
     - `candidate_pairs.tsv` is exactly the scored set, and matches ⊆ candidates
     - the validator passes with `--check-ids`
3. **Write feedback for Claude Code** in `claude/STATUS.md` under a `## Monitor notes` section, as clear numbered instructions: what's wrong, what to rerun, and what's next.
4. **Escalate to the human (Aryan)** for:
   - every submission request (he uploads from one laptop)
   - the French alias review file (`claude/review/alias_candidates.tsv`)
   - the EXP-A decision if the orphan drop is borderline
   - any rule doubt: licenses, external data, whether a feature leaks country
5. **Keep a short running summary** in `claude/research_log.md`:
   - time
   - what was learned
   - current best numbers
   - decisions and their reasons
6. **Watch the clock.** If any step runs more than 2 h over its estimate, cut scope. Drop G first, then the embedding blocker, then C. Never touch the freeze and packaging deadlines.

## 6. Out of scope (don't re-open)
Hungarian/flow, EM/BBSE/GMM/Youden thresholds, Noisy Student, test-time adaptation, bigger cross-encoders, LLM judges, clustering (HAC/Leiden), Fellegi–Sunter LLR features, focal loss, coordinate/phone/geo features, external data of any kind, country as a feature.

**Start now:** read the files in the order at the top, confirm HANDOFF.md is in the repo and Claude Code has started §0, and write your first entry in `claude/research_log.md`.