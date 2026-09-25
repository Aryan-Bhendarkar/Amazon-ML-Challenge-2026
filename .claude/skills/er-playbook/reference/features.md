# Pair feature catalogue (all country-agnostic)

Compute with rapidfuzz vectorized elementwise functions: `rapidfuzz.process.cpdist(a, b, scorer=fuzz.token_set_ratio, workers=-1)`. `a` and `b` are aligned arrays of strings for the pairs, and the output is float32.

## Name (views: n_full, n_core, n_compact, n_alias; also raw lowercased)
- fuzz.ratio, partial_ratio, token_sort_ratio, token_set_ratio, WRatio; JaroWinkler.normalized_similarity; Levenshtein.normalized_distance on compact
- char-3gram TF-IDF cosine (reuse the blocking vectorizer), token Jaccard, IDF-weighted token overlap
- `alias_match`: max similarity between the S1 core and the candidate alias (and vice versa)
- kind flags: cand is a domain/handle/empty name; script is non-Latin (S1 vs cand)

## Difference features (key for hard negatives)
- `extra_tokens` = cand core tokens not in S1 core (fuzzy: no S1 token with JW > 0.9). Record the count, the sum of IDF, the max IDF, and whether any extra token is in a small hand list of business words (holdings, partners, group, exports, services, enterprises, international, global, industries, solutions, center/centre, associates, trading, ventures, systems…).
- `missing_tokens` = S1 core tokens not in cand core. Record the same stats.
- legal forms: equal / both present but different / one missing / both missing (as ordinal codes, not one-hot per form)
- length ratios of core names; first-token equality; last-token equality

## Numbers (house numbers are the #1 decisive signal and the #1 noise source)
- `house_rel`: 0 equal · 1 one is a suffix of the other (407 vs 07) · 2 one is a prefix · 3 leading zeros (4 vs 004) · 4 a digit of edit distance 1 · 5 numeric diff ≤ 10 · 6 different · 7 missing on one side · 8 missing on both
- `house_absdiff` (log1p), `house_same_parity`
- secondary numbers (unit/PMB/door parts): Jaccard of the `a_numbers` sets excluding the house; any conflict
- postcode equal / conflict / missing

## Address
- a_full: token_set_ratio, ratio; a_street: token_set_ratio, JW
- first street token equal; state equal / conflict / missing; n_parts diff
- empty address flags (S1 side never empty; cand side about 3.4%)

## Competition / context (compute after a stage-1 model, or from similarity scores for stage 1)
- rank of this cand among the S1's candidates (by name sim, by stage-1 prob); gap to the S1's best
- rank of this S1 among all S1s retrieving this cand; margin to the second-best S1; n_s1 retrieving the cand
- n_candidates of the S1; the S1's max prob; count of cands above 0.5
- S2↔S3 agreement: max similarity of this cand to the S1's other high-prob cands from the other source

## Leakage guards
- IDF / token frequencies: compute on the records of the split itself (train or test). Never compute them from labels.
- Never compute target statistics (e.g. P(match | token)) that include fold-0 entities.
- `source` (2/3) is OK as a feature. `country` is NOT.
