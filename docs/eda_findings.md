# EDA findings (verified on the full data, 25 Sep)

## Sizes
| split | S1 | S2 | S3 |
|---|---|---|---|
| train | 2,206,821 (US 1.32M / India 0.88M) | 5,034,616 (US 3.02M / IN 2.02M) | 5,285,603 (US 3.17M / IN 2.12M) |
| test | 1,732,544 (India 810k / US 663k / **France 259k**) | 4,887,273 (FR 703k) | 5,082,316 (FR 732k) |

- There are empty addresses: S2/S3 about 3.4%, S1 0%. `<NULL>` appears inside addresses (35k in test S2).
- Some business names are literally `NA` (about 100 in test, mostly France). The loaders keep them as strings.
- About 800 test lines have CSV-quoted fields with doubled quotes. pyarrow/pandas default quoting parses them correctly.

## Ground truth structure
- Matches per S1: 0: 123,247 (5.6%) · 1: 119k · 2: 375k · 3: 531k · 4: 484k · 5: 322k · 6: 165k · 7: 64k · 8: 18.7k · 9: 4.2k · 10: 534 · 11: 37.
- Matched records: S2 3.69M (73% of S2), S3 3.94M (75% of S3). **No record matches more than one S1.**
- So about 27% of S2 and 25% of S3 are distractors.

## Hard negatives (distractors) — examples
| distractor | nearest S1 | decisive difference |
|---|---|---|
| Heritage Metropolitan Biologics **Partners** Inc, 32 GREENWOOD RD, SUDBURY, MA | Heritage Metropolitan Biologics Inc., 31 Greenwood Road, Sudbury, MA | extra word + house no. ±1 |
| Strategic Brown Inc, 4346 SHADOW ROAD, KINGMAN, AZ | Strategic Brown LLC, 4335 Shadow Road, Kingman, AZ | house no. |
| Anurag Academy **Holdings**, SHOP NO-G/N/R … SEVOKE ROAD, SILIGURI | Anurag Academy, Siliguri, Shop No-G/N/R … Sevoke Road | extra word only |
| Accord Cónsultancy **Exports**, 1-13-183/40/**211**, SBI CENTRE | Accord Consultancy, 1-13-183/40/**198**, Sbi Centre | extra word + unit no. |

## Positive noise examples (true matches)
- Numbers: 407 Fleming St ↔ "07 FLEMING ST"; 4 Rock Place ↔ "004 ROCK PL" / "4B Rock Pl"; 298 Eden Lane (a true match with a different number!).
- Names: "energyvrtextile.com", "@SORCHASPIZZA", "SORCHA'S PTA #53907", "Deltatavo formerly known as Niex Holding Company Center", "Cure Hair Enterprises" (with no address!), "Dl Private Limited [Service]", "Sri Dl Advisors Private Limited", Devanagari/Gujarati/Tamil versions of the full name.
- Addresses: city variants (Stanford ↔ Stanfordville Township, Gahanna ↔ Collumbus), states in native script (தமிழ்நாடு), missing house numbers.

## France (test only)
- Few cities: Lille, Nantes, Bordeaux, Calais, Tourcoing, Roubaix, Dunkerque, Pessac, Mérignac, Saint-Nazaire, La Baule, Pornic, Lège-Cap-Ferret, La Teste-de-Buch. That means huge blocks, so city-only blocking is useless there.
- Names are generic and repetitive ("Association de …", "Amicale", "Club", "& Frères", "Ecole"). Expect a higher false-merge risk.
- Legal forms: SARL, SAS, SASU, EURL, SA, SCI, SNC, "S.A.R.L.", "S.A.S.". "(France)" insertions. Accents injected (Fôyer, Spôrtive).
- Addresses: `R.`/`R`/`RUE`, `BD.`, `ALL.`, `Av`, `N°`, `Bis`, `2EME ETAGE`. Region (Nouvelle-Aquitaine) ↔ department (Gironde) swaps. Concatenated handles ("CORALIEFETESSASCOM").

## Samples on disk (after prepare_data.py)
- `data/samples/train_clusters.txt`: 400 GT clusters
- `data/samples/train_unmatched_distractors.tsv`
- `data/samples/test_france.tsv`
