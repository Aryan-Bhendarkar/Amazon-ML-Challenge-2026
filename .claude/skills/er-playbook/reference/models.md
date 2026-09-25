# Model recipes

## LightGBM stage-1 (main workhorse)
- Training pairs: all blocking candidates of ~200–400k S1s from folds 1–4 (labels from GT). Keep all positives. Downsample negatives with sim < a low bar if needed, but **keep the hard ones**.
- Early stopping on a held-out slice of *train folds* (e.g. fold 1), never on fold 0.
- Start params:
  ```python
  dict(objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=200,
       feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
       max_bin=255, n_estimators=3000, verbose=-1, seed=42)
  ```
- Calibrate with isotonic regression on OOF predictions of a train fold. Calibration matters for expected-F0.5 and for a stable threshold.
- Track feature importance (gain). Drop features with zero gain, and drop those that fail LOCO.

## Stage-2 (context)
Same model family, trained on OOF stage-1 probs + context features (ranks, margins, agreement). This is usually a clear precision gain in ER competitions.

## Bi-encoder (retrieval + feature)
- License-safe checkpoints: `intfloat/multilingual-e5-small` (MIT, 118M), `intfloat/multilingual-e5-base` (MIT, 278M), `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (Apache-2.0), `BAAI/bge-m3` (MIT, 568M; too heavy for the laptop). ALWAYS re-check the model card license before use and write it in the docs.
- sentence-transformers, MultipleNegativesRankingLoss + 1 hard negative per pair (mined from blocking). max_seq_length 64, fp16, batch 128 fits 4 GB VRAM for e5-small. 1 epoch over ~1–2M pairs.

## Cross-encoder (precision on the uncertain band)
- Laptop: `microsoft/Multilingual-MiniLM-L12-H384` (MIT) or `xlm-roberta-base` (MIT). SageMaker: `microsoft/mdeberta-v3-base` (MIT).
- Input: `"{name1} | {addr1}" [SEP] "{name2} | {addr2}"`, max_len 96–128, binary CE loss, lr 2e-5, 1 epoch over ~1–3M pairs (balanced hard negatives), fp16.
- Only score pairs with stage-1 prob in [0.05, 0.95] (typically < 20% of pairs). Feed its logit to stage-2.

## LLM (parking lot)
`Qwen2.5-7B-Instruct` (Apache-2.0, 7.6B ≤ 8B) as a judge for the hardest France pairs only. It's expensive, so try it only if everything else is done.
