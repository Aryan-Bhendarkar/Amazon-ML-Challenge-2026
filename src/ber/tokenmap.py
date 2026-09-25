"""Learned token map for native-script names (Phase 1.4).

anyascii turns native-script names into lossy phonetic Latin ('gret phaumdesn' for 'great foundation',
'praibhet' for 'private'). From TRAIN GT pairs (folds 1-4 only: no eval labels) where the candidate
name is non-Latin, align the transliterated tokens with the S1's Latin tokens POSITIONALLY (98% of
such pairs have equal token counts) and keep src -> dst when support >= min_support and the dst
share among src's alignments >= min_precision. The map is applied only to names whose script is
non-Latin (ber.normalize.normalize_name(..., token_map=...)), so Latin records (US, France) are
untouched and no country is ever assumed.

    pairs = aligned_pairs(cand_full, s1_full)        # token-level (src, dst) rows
    tmap, stats = learn(pairs)                        # dict src -> dst
    save(tmap, path); load(path)
"""
from __future__ import annotations

import json
from pathlib import Path

import polars as pl


def aligned_pairs(cand_full: pl.Series, s1_full: pl.Series) -> pl.DataFrame:
    """Token-level (src, dst) alignments of equal-token-count name pairs (normalized n_full strings)."""
    d = pl.DataFrame({"c": cand_full.str.split(" "), "s": s1_full.str.split(" ")})
    d = d.filter((pl.col("c").list.len() == pl.col("s").list.len()) & (pl.col("c").list.len() > 0))
    return d.explode(["c", "s"], empty_as_null=True).rename({"c": "src", "s": "dst"}).filter(
        (pl.col("src") != "") & (pl.col("dst") != ""))


def learn(pairs: pl.DataFrame, min_support: int = 3, min_precision: float = 0.5) -> tuple[dict, dict]:
    """pairs: (src, dst) rows. Returns ({src: dst}, stats). Identity majorities are not stored."""
    cnt = pairs.group_by("src", "dst").len()
    tot = cnt.group_by("src").agg(pl.col("len").sum().alias("n_src"))
    best = (cnt.sort(["src", "len", "dst"], descending=[False, True, False]).group_by("src", maintain_order=True)
               .first().join(tot, on="src").with_columns((pl.col("len") / pl.col("n_src")).alias("prec")))
    keep = best.filter((pl.col("len") >= min_support) & (pl.col("prec") >= min_precision) & (pl.col("src") != pl.col("dst")))
    tmap = dict(zip(keep["src"].to_list(), keep["dst"].to_list()))
    covered = best.filter((pl.col("len") >= min_support) & (pl.col("prec") >= min_precision))["len"].sum()
    stats = {"n_token_alignments": int(pairs.height), "n_src_types": int(tot.height), "n_map": len(tmap),
             "alignments_covered_by_rules": float(covered / max(1, pairs.height)),
             "min_support": min_support, "min_precision": min_precision}
    return tmap, stats


def apply(tokens: list[str], tmap: dict) -> list[str]:
    return [tmap.get(t, t) for t in tokens]


def save(tmap: dict, path: Path, meta: dict | None = None) -> None:
    Path(path).write_text(json.dumps({"meta": meta or {}, "map": tmap}, ensure_ascii=True, sort_keys=True),
                          encoding="utf-8")


def load(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))["map"]
