# Blocking recipes

Always block **within country** (the country string as a partition key is fine; just never assume the set of values). Evaluate with `ber.blocking_eval.blocking_report` on `mini`, where queries are eval S1s and the pool is ALL train S2+S3 of that country.

## 1. Exact keys (cheap, high precision; recall ~0.8–0.9 alone)
```python
# norm cache columns: n_core, n_compact, a_house, a_street, country
k_addr = country + "|" + a_house + "|" + first_street_token[:4]          # skip if a_house == ""
k_name = country + "|" + tok1 + "|" + tok2[:3]                           # tokens of n_core, len(tok1)>=3
k_comp = country + "|" + n_compact[:8]                                   # domains / handles / squashed names
k_alias = country + "|" + alias_compact[:8]                              # "formerly known as" names
```
Join S1 keys to S2/S3 keys with pandas/polars merges. **Cap blocks**: drop keys with > 300 records on the S2/S3 side (generic tokens like "association", "shree", "global"), and count how much recall the cap loses. Prefer 2 tokens over caps where possible.

## 2. Char n-gram TF-IDF top-k (workhorse for typos, reordering, transliteration)
```python
from sklearn.feature_extraction.text import TfidfVectorizer
vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), min_df=2, dtype=np.float32, sublinear_tf=True)
vec.fit(pd.concat([s1.n_core, pool.n_core]))          # unsupervised fit on the split itself: allowed
A = vec.transform(s1.n_core); B = vec.transform(pool.n_core).T.tocsr()
for chunk in range(0, A.shape[0], 2000):              # sparse matmul in chunks, keep top-k per row
    S = A[chunk:chunk+2000] @ B                        # csr (2000 x n_pool)
    # top-k per row from S.data/S.indptr (argpartition per row), k ≈ 30
```
- Use `sparse_dot_topn` (Apache-2.0) for a fast multithreaded top-k sparse matmul if the pip install works; otherwise use the chunked numpy loop above.
- Do this per country. For US (≈6M pool) keep chunks small, or sub-partition by the state token when present, with a fallback pass without state for records lacking it.
- Name-only TF-IDF confuses chains/generic names. Add a second TF-IDF over `n_core + " " + a_street` (name+street), and union the top-k of both.

## 3. Dense retrieval (native script, heavy noise)
- Fine-tune `intfloat/multilingual-e5-small` (MIT) or `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (Apache-2.0) with MultipleNegativesRankingLoss on (S1 text, matched record text) pairs from folds 1–4. Text = `"name | address"` (raw, not transliterated, so the model learns scripts).
- Embed at 384-d fp16. For FAISS: `IndexHNSWFlat` (CPU) or `IndexIVFFlat` per country. top-k = 20.

## 4. Reverse, record-centric retrieval
For each S2/S3 record, retrieve the top-3 S1s (same retrievers, roles swapped). Union into the candidate set. It fits the at-most-one-owner structure, recovers matches that an S1's top-k missed because of competition, and supplies "rank of this S1 for the record" features.

## Accounting
Report per retriever:
- recall alone
- marginal recall added to the union
- pairs added

Keep a retriever if its marginal recall per million pairs is worth it. The final union is **exactly** what the model scores, and it becomes candidate_pairs.tsv.
