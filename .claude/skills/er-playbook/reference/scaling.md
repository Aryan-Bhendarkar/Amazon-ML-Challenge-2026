# Scaling recipes (16 GB RAM, 12 threads, 4 GB VRAM laptop; 10M records)

- **Integer ids**: map entity_id → int32 row index once per split. Pair tables hold `(s1_idx:int32, cand_idx:int32)` = 8 bytes per pair. 40M pairs = 320 MB.
- Load only the needed columns: `ber.io.load_source(split, s, columns=[...])`. The normalized cache is parquet, so column pruning is free.
- **Process per country**, and for the US per state-block or per chunk of S1 queries. Free memory between chunks (`del`, `gc.collect()`).
- Strings for features: gather aligned arrays with `np.take(names, idx)` (object arrays), then run rapidfuzz `cpdist(..., workers=-1)`. It is C++ and multithreaded: roughly 5–20M pairs/s for simple scorers.
- Prefer polars for big joins/group-bys (multithreaded, lower memory). Convert to pandas only for small frames.
- float32 everywhere. LightGBM: `max_bin=255`, construct the Dataset once and save a binary (`ds.save_binary`).
- Windows multiprocessing: always put `if __name__ == "__main__":` in the script; workers get pickled data, so pass paths/indices, not huge frames.
- GPU 4 GB: fp16, max_len 64, batch 128 for MiniLM/e5-small; `torch.cuda.empty_cache()` between phases. Anything larger goes to SageMaker g5.
- Time budget heuristics: normalization of 22M records ≈ 3–5 min with 10 processes. TF-IDF top-30 for 88k queries vs a 3M pool ≈ minutes. For the full 1.7M test queries, estimate first with a 1% run and extrapolate.
- Always log `time.time()` deltas and peak RSS (`psutil.Process().memory_info().rss`) for long steps into run metrics.
- Windows: `import torch` first in any script using torch (DLL load-order OSError otherwise). Run long jobs in the background with logs under `logs/`.
