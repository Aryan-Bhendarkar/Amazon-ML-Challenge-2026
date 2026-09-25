"""One-time data preparation (idempotent; ~2-4 min on a laptop).

  1. raw TSVs -> parquet (data/parquet/{split}_s{n}.parquet), row counts checked vs line counts
  2. GT -> exploded pairs (data/parquet/train_gt_pairs.parquet)
  3. folds for train S1 (data/splits/folds.parquet)  [see src/ber/split.py]
  4. human-readable samples (data/samples/) for eyeballing noise patterns

Usage:  python scripts/prepare_data.py [--force]
"""
import _bootstrap  # noqa: F401

import argparse
import random
import time

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ber import io, paths, split


def count_lines(p) -> int:
    n = 0
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            n += chunk.count(b"\n")
    return n


def main(force: bool) -> None:
    paths.ensure_dirs()
    t0 = time.time()
    for sp in paths.SPLITS:
        for s in paths.SOURCES:
            out = paths.parquet_path(sp, s)
            if out.exists() and not force:
                print(f"skip {out.name} (exists)")
                continue
            df = io.read_source_tsv(sp, s)
            expect = count_lines(paths.raw_tsv(sp, s)) - 1
            assert len(df) == expect, f"{sp} s{s}: parsed {len(df)} rows but file has {expect} lines"
            assert df.entity_id.is_unique, f"{sp} s{s}: duplicate entity_id"
            assert df.entity_id.str.startswith(f"S{s}-").all(), f"{sp} s{s}: bad id prefix"
            pq.write_table(pa.Table.from_pandas(df, preserve_index=False), out, compression="zstd")
            print(f"{out.name}: {len(df):,} rows  countries={df.country.value_counts().to_dict()}  "
                  f"({time.time() - t0:.0f}s)")

    gt = io.read_gt_tsv()
    s1 = io.load_source("train", 1, columns=["entity_id", "country"])
    assert len(gt) == len(s1) and set(gt.source1_entity_id) == set(s1.entity_id), "GT/S1 id mismatch"
    if force or not paths.gt_pairs_path().exists():
        pairs = io.gt_to_pairs(gt)
        assert not pairs.match_id.duplicated().any(), "a record matches >1 S1 (assumption broken!)"
        pq.write_table(pa.Table.from_pandas(pairs, preserve_index=False), paths.gt_pairs_path())
        print(f"gt pairs: {len(pairs):,}")
    if force or not paths.folds_path().exists():
        folds = split.make_folds(gt, s1)
        pq.write_table(pa.Table.from_pandas(folds, preserve_index=False), paths.folds_path())
        print("folds:", folds.fold.value_counts().sort_index().to_dict(),
              "| mini:", int(folds["mini"].sum()), "| micro:", int(folds["micro"].sum()))
        print("fold0 by country:", folds[folds.fold == 0].country.value_counts().to_dict())

    # samples for humans / Claude to eyeball
    smp = paths.SAMPLE_DIR / "train_clusters.txt"
    if force or not smp.exists():
        rng = random.Random(0)
        pick = set(rng.sample(list(gt.source1_entity_id), 400))
        all_pairs = io.load_gt_pairs()
        pairs = all_pairs[all_pairs.s1_id.isin(pick)]
        need = sorted(pick | set(pairs.match_id))
        rec = pd.concat([pq.read_table(paths.parquet_path("train", s), filters=[("entity_id", "in", need)]).to_pandas()
                         for s in paths.SOURCES]).set_index("entity_id")
        with open(smp, "w", encoding="utf-8", newline="\n") as f:
            for s1id in sorted(pick):
                r = rec.loc[s1id]
                f.write(f"### {s1id} | {r.business_name} | {r.business_address} | {r.country}\n")
                for m in pairs.loc[pairs.s1_id == s1id, "match_id"]:
                    r = rec.loc[m]
                    f.write(f"    {m} | {r.business_name} | {r.business_address}\n")
        s2 = io.load_source("train", 2)
        unmatched = s2[~s2.entity_id.isin(set(all_pairs.match_id))]
        del s2
        unmatched.sample(300, random_state=0).to_csv(paths.SAMPLE_DIR / "train_unmatched_distractors.tsv",
                                                     sep="\t", index=False, lineterminator="\n")
        test = io.load_all_sources("test", countries=["France"])
        test.groupby("source").sample(150, random_state=0).to_csv(
            paths.SAMPLE_DIR / "test_france.tsv", sep="\t", index=False, lineterminator="\n")
        print("samples written to", paths.SAMPLE_DIR)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    main(ap.parse_args().force)
