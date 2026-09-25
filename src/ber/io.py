"""Data loading. ALWAYS load data through these functions.

Pitfalls these loaders handle (see docs/rules.md):
  * files are TAB separated; some fields are CSV-quoted with doubled quotes ("" -> ")
  * literal business name "NA" exists (mostly France) -> must NOT become NaN
  * empty matched_entity_ids in GT -> empty list, not NaN / "nan"
  * IDs are strings (not zero padded) -> never cast to int
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from . import paths

SOURCE_COLS = ["entity_id", "business_name", "business_address", "country"]


def _read_tsv_arrow(path: Path, columns: list[str]) -> pa.Table:
    return pacsv.read_csv(
        path,
        read_options=pacsv.ReadOptions(use_threads=True, block_size=1 << 26),
        parse_options=pacsv.ParseOptions(delimiter="\t", quote_char='"', double_quote=True,
                                         newlines_in_values=False),
        convert_options=pacsv.ConvertOptions(
            column_types={c: pa.string() for c in columns},
            strings_can_be_null=False,      # keeps "NA", "NULL", "" as literal strings
            include_columns=columns,
        ),
    )


def read_source_tsv(split: str, source: int) -> pd.DataFrame:
    """Read a raw source TSV (slow path). Prefer load_source() which uses the parquet cache."""
    t = _read_tsv_arrow(paths.raw_tsv(split, source), SOURCE_COLS)
    return t.to_pandas()


def read_gt_tsv() -> pd.DataFrame:
    """Raw GT: one row per train S1 id, matched_entity_ids as comma string ('' when none)."""
    t = _read_tsv_arrow(paths.raw_gt(), ["source1_entity_id", "matched_entity_ids"])
    return t.to_pandas()


def gt_to_pairs(gt: pd.DataFrame) -> pd.DataFrame:
    """Explode GT into (s1_id, match_id) pairs. S1 ids with no matches are dropped here —
    use load_gt_sets() or the folds table when you need singletons."""
    s = gt.set_index("source1_entity_id")["matched_entity_ids"]
    s = s[s.str.len() > 0].str.split(",").explode()
    out = s.reset_index()
    out.columns = ["s1_id", "match_id"]
    return out


def load_source(split: str, source: int, countries: Iterable[str] | None = None,
                columns: list[str] | None = None) -> pd.DataFrame:
    """Load one source from the parquet cache (run scripts/prepare_data.py first).
    countries: optional filter, e.g. ["India"]. Never hard-code a country list for test —
    test contains France (unseen in train)."""
    p = paths.parquet_path(split, source)
    if not p.exists():
        raise FileNotFoundError(f"{p} missing — run: python scripts/prepare_data.py")
    filters = [("country", "in", list(countries))] if countries is not None else None
    return pq.read_table(p, columns=columns, filters=filters).to_pandas()


def load_all_sources(split: str, countries: Iterable[str] | None = None,
                     columns: list[str] | None = None) -> pd.DataFrame:
    """S1+S2+S3 of a split stacked, with an int8 `source` column (1/2/3)."""
    parts = []
    for s in paths.SOURCES:
        df = load_source(split, s, countries, columns)
        df["source"] = pd.Series(s, index=df.index, dtype="int8")
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def load_gt_pairs() -> pd.DataFrame:
    return pq.read_table(paths.gt_pairs_path()).to_pandas()


def load_gt_sets(s1_ids: Iterable[str] | None = None) -> dict[str, set[str]]:
    """{s1_id: set(match_ids)} INCLUDING singletons (empty set). Restrict with s1_ids."""
    folds = load_folds()
    if s1_ids is not None:
        folds = folds[folds["s1_id"].isin(set(s1_ids))]
    out: dict[str, set[str]] = {k: set() for k in folds["s1_id"]}
    pairs = load_gt_pairs()
    pairs = pairs[pairs["s1_id"].isin(out.keys())]
    for s1, m in zip(pairs["s1_id"].to_numpy(), pairs["match_id"].to_numpy()):
        out[s1].add(m)
    return out


def load_folds() -> pd.DataFrame:
    """Columns: s1_id, country, n_matches, fold (0..K-1), mini (bool dev subset of fold 0)."""
    p = paths.folds_path()
    if not p.exists():
        raise FileNotFoundError(f"{p} missing — run: python scripts/prepare_data.py")
    return pq.read_table(p).to_pandas()


def source_of(entity_id: str) -> int:
    return int(entity_id[1])
