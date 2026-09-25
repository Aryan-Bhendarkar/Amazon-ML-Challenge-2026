# France: the unseen-country problem (15% of test S1 ≈ 0.15 of the final score)

## What we know (test data inspection)
- About 259k S1 across ~15 cities, so blocks are dense. Names are generic ("Association de …", "Club", "Amicale", "& Frères").
- Legal forms: SARL, SAS, SASU, SA, EURL, SCI, SNC, SCP, SELARL; dotted forms ("S.A.R.L.").
- Noise looks like the same generator as US/India:
  - "(France)" insertion, accents injected, handles ("CORALIEFETESSASCOM")
  - `N°`, `Bis`/`Ter`, `R.`/`BD.`/`ALL.`/`AV`, "2EME ETAGE"
  - region ↔ department swaps (Nouvelle-Aquitaine ↔ Gironde; Pays de la Loire ↔ Loire-Atlantique; Hauts-de-France ↔ Nord / Pas-de-Calais)

## Strategy
1. **Transferable model**: only similarity/difference/number-relation/rank features. No `country` feature, and no features whose meaning depends on country (e.g. a raw 'state' code equality is fine; a feature like 'is_india' is not).
2. **LOCO validation as the proxy**: train on US only and eval India; train on India only and eval US. Keep features/models that keep LOCO high. Report the LOCO F0.5 for every model change.
3. **French normalization** (hand-written general knowledge, allowed):
   - rue/r/r. → rue; boulevard/bd/bd. → blvd; avenue/av/av. → ave; allée/all/all. → allee; impasse/imp → imp; chemin/ch → ch; route/rte → rte; place/pl → pl; faubourg/fg → fg; quai; cours; square
   - n°/no → drop; bis/ter/quater → attach to the number (53 bis → 53b)
   - saint/st/ste consistent with generic rules
   - legal forms canonical; "(france)" → drop as noise
   - region ↔ department map for the cities in test (hand-written)
4. **Synthetic French training pairs**: take test-France S1 records (no labels used) and apply noise operators estimated from train pairs:
   - case change, legal-form swap/drop, token drop/insert, typo, accent injection, reorder, abbreviation, address component reorder, number noise
   - hard negatives: + business word, house ±k

   Fine-tune / add to training. It is unsupervised use of test inputs, so document it clearly in the methodology.
5. **Conservative decision for France** (see decision.md) if LOCO shows threshold drift.
6. Diagnostics on every test run: France empty-rate and mean matches should look like US/India. Big deviations signal a pipeline problem (e.g. blocking collapse in dense cities).
