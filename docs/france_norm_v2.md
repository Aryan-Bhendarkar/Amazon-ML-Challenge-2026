# norm v2: French normalization + legal forms (Lane C, 26 Sep 2026)

Code: `src/ber/normalize.py` (`COUNTRY_NAME_RULES`, `_FR_REGION_DEPTS`/`FR_STATES`, `_FR_NUM_SUFFIX`),
`scripts/build_norm_cache.py` (NORM_VERSION 2 passes `country` to `normalize_name`), tests in
`tests/test_normalize.py`. All rules are **hand-written general knowledge** (no external data) and
**country-keyed with a generic fallback** (an unknown country gets no rule).

## What changed
| Field | Rule (France only) | Why (label-free evidence, test S1 vs pool) |
|---|---|---|
| address `state` | 13 regions (2016) + pre-2016 region names + 96 departments → region code (`pdl`, `hdf`, `naq`, …), matched on a whole comma component like US/IN states; removed from `a_street` | every FR test S1 ends with a region; the pool has the region in 33%, the **department** in 31% (`Loire-Atlantique`, `Gironde`, `Nord`, `Pas-de-Calais`), neither in 36%. v1 left these as unmatched street tokens and `state_rel` = missing for 100% of FR pairs |
| house number | `1bis`, `15 bis`, `3 ter`, `quater` → the number only | 5% of FR S1 addresses; v1 parsed `1BIS` as house `1b` vs `1` |
| name | strip `(France)` | 8.0% of FR S1 names, 6.5% of pool names; "france" is the #8 single extra token among same-address pairs |
| name tokens | `cie`/`compagnie` → `co` (legal form, like `company`); `et` → `and`; `frs` → `freres`; `st` → `saint`; `5arl`/`5as`/`5asu` → `sarl`/`sas`/`sasu` | swap counts among S1–pool pairs sharing house + street: cie↔compagnie 503, and↔et 384, freres↔frs 232, saint↔st 155, 5arl/5as↔sarl/sas 243 |
| legal form | `ei` (entreprise individuelle) | #7 single extra/missing token (755) among same-address pairs |

`cie` (86), `ei` (455) and `5as` (8) occur in US/India **train** names, so they are France-keyed as well
rather than global: that keeps the train normalization byte-identical and the shipped model valid.

## Neutrality on train / US / India (exact, not sampled)
- `norm_v2` vs `norm_v1`, all 6 cache files: **0 changed rows in train**; on test **only France rows change**
  (`pipelines/refeat_norm.changed_countries`). Hence LOCO and clean val are *identical* by construction
  (LOCO Δ = 0, clean Δ = 0): no model or train feature changes.
- `pipelines/refeat_norm.py --check` (recompute base features from norm frames for 190k US/India mini
  pairs): every feature column equals the cached value, so the France re-featurization path is faithful.

## Label-free France test-pair change report (20k random FR test S1, 1.87M candidate pairs of v1_n2)
| statistic | FR v1 | FR v2 | US (v1) | India (v1) |
|---|---|---|---|---|
| S1 with ≥1 candidate at name_tset ≥ 90 **and** addr_tset ≥ 90 | 0.908 | **0.976** | 0.961 | 0.968 |
| pairs with name ≥ 90 and addr ≥ 90 | 0.029 | 0.040 | 0.037 | 0.035 |
| pairs with addr_tset ≥ 90 | 0.181 | 0.242 | 0.080 | 0.101 |
| likely-match slice (name ≥ 95, same numbers): addr_tset ≥ 90 | 0.578 | 0.737 | 0.892 | 0.895 |
| likely-match slice: state same / diff / missing | 0 / 0 / 1.00 | 0.64 / 0.03 / 0.33 | 0.98 / 0.02 / 0 | 0.75 / 0.04 / 0.22 |

- 20% of FR pairs change `name_tset` (11.5% up, 8.8% down), 42% change `addr_tset` (22% up, 20% down:
  the region/department token now counts as a state, not a street word; pairs across regions lose it).
- France moves from an outlier (0.908) into the US/India regime (0.96–0.98) on "has a strong candidate".
- The remaining likely-slice address gap (0.74 vs 0.89) is mostly **genuinely different streets** (same
  generic name + same number, e.g. `30 blvd jules simon bordeaux` vs `30 rue des muniers lille`): the
  templated-twin negatives, not normalization. `state_rel = diff` (3%) now flags cross-region pairs,
  a strong learned negative in US/IN.

## Test-pool statistics (unsupervised, `SplitContext`, S1 level)
| split / country | S1 name key shared | S1 address shared | max token df/n_s1 (median) | S1 with a token in > 2% of S1 | addr key with no pool record |
|---|---|---|---|---|---|
| train India | 0.544 | 0.066 | 0.0103 | 0.161 | 0.51 |
| train US | 0.480 | 0.072 | 0.0046 | 0.162 | 0.32 |
| test India | 0.536 | 0.064 | 0.0102 | 0.160 | 0.51 |
| test US | 0.401 | 0.053 | 0.0046 | 0.163 | 0.32 |
| **test France v1** | 0.525 | **0.191** | **0.0267** | **0.662** | 0.45 |
| **test France v2** | 0.566 | **0.198** | **0.0257** | **0.607** | 0.19 |

France is 3× more co-located (G2 inputs) and has 3–6× denser generic name tokens (G3 lfrac inputs) than
anything in train, while US and India look alike on both. **LOCO (US↔India) cannot see these two shifts**,
so a LOCO-neutral G2/G3 is not evidence of France safety; see the Lane C audit for the coarsening decision.

## How to apply on test (test lane, not box2)
1. `norm_v2_test_s{1,2,3}.parquet` (and train, identical to v1): `python scripts/build_norm_cache.py --split test`
   (≈ 6 min with 2 jobs).
2. `python pipelines/refeat_norm.py --cache v1_n2 --tag test` → `data/cands/v1_n2/test_n2.parquet` (France
   pairs re-featurized, ≈ 24M pairs; US/India rows copied; candidate set unchanged).
3. `python pipelines/predict_test_v1.py --run <run> --cache v1_n2 --src-tag test_n2 --norm 2 --out-suffix _n2`
   (ctx statistics on norm_v2; saves `test_feats_g15_ctx3_test_n2_n2.parquet`, never overwrites the default
   matrix). A cheaper variant is to recompute ctx only for France rows and take US/India rows from the
   existing matrix (identical inputs).
