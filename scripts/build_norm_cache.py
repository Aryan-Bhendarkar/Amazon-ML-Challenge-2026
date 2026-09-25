"""Normalize all records once and cache (data/features/norm_v{V}_{split}_s{n}.parquet).

Streams parquet batches -> worker pool -> incremental ParquetWriter, so peak RAM stays low
(~1-2 GB) even for 5M-row sources on a 16 GB laptop.
Bump NORM_VERSION when normalization logic changes, so old caches are never mixed with new code.

Usage:  python scripts/build_norm_cache.py [--split train|test|all] [--jobs 6] [--limit N]
"""
import _bootstrap  # noqa: F401

import argparse
import os
import time
from multiprocessing import Pool

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ber import paths, tokenmap
from ber.normalize import normalize_address, normalize_name, script_of

# v0: hand rules only. v1: + learned native-script token map (scripts/build_token_map.py, folds 1-4 pairs)
NORM_VERSION = 1
TOKEN_MAP_PATH = paths.FEATURE_DIR / "token_map_v1.json"
_TMAP = None


def _init(tmap):
    global _TMAP
    _TMAP = tmap
NAME_COLS = ["n_full", "n_core", "n_legal", "n_compact", "n_alias", "n_kind"]
ADDR_COLS = ["a_full", "a_street", "a_numbers", "a_house", "a_postcode", "a_state", "a_nparts", "a_empty"]
BATCH = 100_000


def norm_path(split: str, source: int):
    return paths.FEATURE_DIR / f"norm_v{NORM_VERSION}_{split}_s{source}.parquet"


def _work(rows: dict) -> pa.Table:
    names = [normalize_name(x, token_map=_TMAP) for x in rows["business_name"]]
    addrs = [normalize_address(a, c) for a, c in zip(rows["business_address"], rows["country"])]
    out = pd.DataFrame(names, columns=NAME_COLS)
    out[ADDR_COLS] = pd.DataFrame(addrs, columns=ADDR_COLS)
    out["a_nparts"] = out["a_nparts"].astype("int16")
    out["a_empty"] = out["a_empty"].astype(bool)
    out["n_script"] = [script_of(x) for x in rows["business_name"]]
    out.insert(0, "entity_id", rows["entity_id"])
    out.insert(1, "country", rows["country"])
    return pa.Table.from_pandas(out, preserve_index=False)


def _batches(split: str, source: int, limit: int | None):
    pf = pq.ParquetFile(paths.parquet_path(split, source))
    seen = 0
    for b in pf.iter_batches(batch_size=BATCH):
        d = b.to_pydict()
        if limit is not None:
            if seen >= limit:
                return
            d = {k: v[: limit - seen] for k, v in d.items()}
        seen += len(d["entity_id"])
        yield d


def main(splits, jobs: int, limit: int | None) -> None:
    paths.ensure_dirs()
    tmap = None
    if NORM_VERSION >= 1:
        if not TOKEN_MAP_PATH.exists():
            raise FileNotFoundError(f"{TOKEN_MAP_PATH} missing — run: python scripts/build_token_map.py")
        tmap = tokenmap.load(TOKEN_MAP_PATH)
    with Pool(jobs, initializer=_init, initargs=(tmap,)) as pool:
        for sp in splits:
            for s in paths.SOURCES:
                t0 = time.time()
                p = norm_path(sp, s) if not limit else paths.FEATURE_DIR / f"_limit_{sp}_s{s}.parquet"
                tmp = p.with_suffix(".tmp")
                writer, n = None, 0
                for tbl in pool.imap(_work, _batches(sp, s, limit)):     # ordered, streaming
                    if writer is None:
                        writer = pq.ParquetWriter(tmp, tbl.schema, compression="zstd")
                    writer.write_table(tbl.cast(writer.schema))
                    n += tbl.num_rows
                if writer is not None:
                    writer.close()
                    os.replace(tmp, p)
                print(f"{p.name}: {n:,} rows in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="all", choices=["train", "test", "all"])
    ap.add_argument("--jobs", type=int, default=max(1, min(8, (os.cpu_count() or 4) - 4)))
    ap.add_argument("--limit", type=int, default=None, help="smoke test on first N rows")
    a = ap.parse_args()
    main(paths.SPLITS if a.split == "all" else (a.split,), a.jobs, a.limit)
