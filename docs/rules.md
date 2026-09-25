# Rules, fine print and gotchas (from the Guideline PDF, the Problem Statement PDF, and the validator code)

## Timeline & submissions
- Window: **25 Sep 2026 00:00 IST → 27 Sep 2026 23:59 IST**. The submit button is disabled afterwards.
- **5 leaderboard submissions/day × 3 days = 15 total.** Assume unused ones do not carry over.
- Keep a version history of every submission (shortlisting is based on submitted solutions). `submissions/records/` does this.
- Public leaderboard = a subset of test; private = the rest. We always predict the full test set. The guideline says "evaluation and shortlisting based on performance across both leaderboards"; the problem statement says final rankings use the private one. So don't overfit public.
- One device per participant, **no simultaneous logins** (the system may terminate the challenge). One teammate and one laptop upload.
- Multiple registrations = disqualification. Queries go through the Google Form; tech issues to support@unstop.com.

## Deliverables
- Leaderboard: `matching_results.tsv` only.
- Final zip `<team_name>_submission.zip`: `output/{matching_results.tsv, candidate_pairs.tsv}`, `code/business_entity_resolution/{src/, README.md, requirements.txt}`, `Documentation_template.md` (filled in, .md or .pdf). It must regenerate both outputs from the train/test data alone. The top teams are audited.
- Documentation: the guideline says 1–2 pages, while the problem statement says there is no limit. We will write 2 tight pages plus an appendix. It must cover the methodology, blocking strategy, model architecture and features, and anything else relevant.
- Top 100 teams are then asked for methodology details; every team member must satisfy the eligibility criteria.

## candidate_pairs.tsv
- It must be the **exact set the final model scores** (the last filtering stage). matching ⊆ candidates.
- It is not scored, but it is used to audit recall ceiling and reduction ratio.

## Model / data rules
- Final model: **MIT or Apache-2.0, ≤ 8B params.** We apply this to ALL pretrained weights.
  - OK: multilingual-e5 (MIT), bge-m3 (MIT), LaBSE (Apache), XLM-R (MIT), mDeBERTa-v3 (MIT), MiniLM variants (check card; many are Apache/MIT), Qwen2.5-7B-Instruct (Apache), Mistral-7B (Apache), LightGBM (MIT), XGBoost (Apache), CatBoost (Apache).
  - NOT OK: Llama*, Gemma*, Qwen2.5-3B/72B (Qwen license), CC-BY-NC models.
- **Prohibited:** entity-resolution APIs, government/business registries, geocoding, and any internet data augmentation (postcode DBs, city lists, gazetteers).
- Hand-written dictionaries of general knowledge are allowed; keep them in code and list them in the documentation.
- `country` is an open set. France is in test only and must be present in the output.

## Metric (per S1, then macro average over ALL S1)
| truth | prediction | score |
|---|---|---|
| empty | empty | 1.0 |
| empty | anything | 0 |
| n>0 | empty | 0 |
| n=3 | 3 correct | 1.000 |
| n=3 | 2 correct | 0.909 |
| n=3 | 1 correct | 0.714 |
| n=3 | 3 correct + 1 wrong | 0.789 |
| n=3 | 2 correct + 1 wrong | 0.714 |

## Format gotchas (validator internals)
- Header must be exactly `source1_entity_id<TAB>matched_entity_ids` (candidates: `candidate_entity_ids`).
- IDs are split on `,` **without stripping** → no spaces. Only `\n` is stripped, so **CRLF corrupts the last ID** (the validator passes it by default!). Empty rows need the tab.
- Rejections:
  - a row without a tab
  - a duplicate S1 row
  - missing or extra S1 IDs
  - an S1 ID inside a list
  - a non-S2/S3 prefix
  - a duplicate ID within a list
  - non-UTF-8
- Nonexistent S2/S3 IDs only lower the score. `--check-ids` verifies them (needs a few GB of RAM).

## Open questions (send via Google Form; record answers here)
1. Which submission counts for the final ranking: the last one, the best public one, or one we select? → _pending_
2. Documentation length limit? → _pending_
3. Does the license/size rule cover all pretrained models or only the final matcher? → _pending_
4. Are hand-written normalization dictionaries OK? → **YES (judge clarification, 26 Sep):** prohibited = external databases, APIs, geocoding/entity lookup, internet-sourced augmentation, and packages bundling external geo/postal/business data (libpostal, geocoders, postal-code or gazetteer datasets). Allowed = pure-algorithm libraries (RapidFuzz, jellyfish, scikit-learn, LightGBM, pandas), general-language pretrained NLP/embedding models within the license/size limits, any algorithm using only the provided records, and SMALL hand-written normalization dictionaries.
5. When does the daily submission count reset? → _pending_
