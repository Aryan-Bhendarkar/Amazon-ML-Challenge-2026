"""Writing leaderboard / final-package files. The ONLY sanctioned way to write outputs.

Guards against every rejection rule + the silent Windows CRLF bug (pandas.to_csv on Windows
writes '\r\n', which glues '\r' onto the last ID of each row -> scored as a wrong ID).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd

from . import paths

MATCH_HEADER = ("source1_entity_id", "matched_entity_ids")
CAND_HEADER = ("source1_entity_id", "candidate_entity_ids")


def write_id_lists(path: Path, lists: Mapping[str, Iterable[str]], s1_order: list[str],
                   header: tuple[str, str]) -> dict:
    """Write one row per S1 in `s1_order` (the test_source1 order). Dedupes IDs, drops anything
    that is not S2-/S3-, writes UTF-8 with '\n' line endings. Returns simple stats."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wanted = set(s1_order)
    extra = set(lists) - wanted
    if extra:
        raise ValueError(f"{len(extra)} S1 ids not in test_source1, e.g. {sorted(extra)[:3]}")
    n_nonempty = n_ids = dropped = 0
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\t".join(header) + "\n")
        for s1 in s1_order:
            seen, ids = set(), []
            for x in lists.get(s1, ()):
                x = str(x).strip()
                if not (x.startswith("S2-") or x.startswith("S3-")):
                    dropped += 1
                    continue
                if x not in seen:
                    seen.add(x)
                    ids.append(x)
            n_nonempty += bool(ids)
            n_ids += len(ids)
            f.write(f"{s1}\t{','.join(ids)}\n")
    return {"rows": len(s1_order), "nonempty": n_nonempty, "ids": n_ids, "dropped_bad_ids": dropped}


def write_candidates_streaming(parquet_path: Path, out_tsv: Path, s1_order: list[str],
                               matched_pairs: set | None = None) -> dict:
    """Stream a (s1_id, cand_id) parquet to candidate_pairs.tsv without loading it all.
    Each parquet ROW GROUP must contain complete S1 groups (write one row group per S1 chunk).
    If matched_pairs is given, also counts how many matched pairs are present (matches ⊆ candidates)."""
    import pyarrow.parquet as pq
    out_tsv = Path(out_tsv)
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    wanted, seen = set(s1_order), set()
    n_ids = found = 0
    pf = pq.ParquetFile(parquet_path)
    with open(out_tsv, "w", encoding="utf-8", newline="\n") as f:
        f.write("\t".join(CAND_HEADER) + "\n")
        for rg in range(pf.num_row_groups):
            t = pf.read_row_group(rg, columns=["s1_id", "cand_id"]).to_pandas()
            for s1, ids in t.groupby("s1_id", sort=False)["cand_id"]:
                if s1 in seen:
                    raise ValueError(f"S1 {s1} split across row groups - write one row group per S1 chunk")
                if s1 not in wanted:
                    raise ValueError(f"S1 {s1} not in test_source1")
                seen.add(s1)
                u = list(dict.fromkeys(x for x in ids if x.startswith(("S2-", "S3-"))))
                n_ids += len(u)
                if matched_pairs is not None:
                    found += sum((s1, x) in matched_pairs for x in u)
                f.write(f"{s1}\t{','.join(u)}\n")
        for s1 in s1_order:
            if s1 not in seen:
                f.write(f"{s1}\t\n")
    out = {"rows": len(s1_order), "nonempty": len(seen), "ids": n_ids}
    if matched_pairs is not None:
        out["matched_pairs_in_candidates"] = found
        out["matched_pairs_total"] = len(matched_pairs)
    return out


def run_validator(matching: Path, candidate: Path | None = None, check_ids: bool = True) -> tuple[bool, str]:
    """Run the official student_resource/utils/validate_submission.py. Returns (passed, output)."""
    cmd = [sys.executable, str(paths.VALIDATOR), "--matching", str(matching),
           "--test-dir", str(paths.RAW_DIR / "test")]
    if candidate is not None:
        cmd += ["--candidate", str(candidate)]
    if check_ids:
        cmd.append("--check-ids")
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    out = r.stdout + r.stderr
    return r.returncode == 0, out


def diagnostics(matches: Mapping[str, Iterable[str]], test_s1: pd.DataFrame) -> dict:
    """Per-country prediction profile on TEST — compare with train priors to catch collapse
    (e.g. France getting 0 matches, or over-merging). Train priors: ~5.6% singletons, ~3.5 matches."""
    df = test_s1[["entity_id", "country"]].copy()
    n = df.entity_id.map(lambda s: len(set(matches.get(s, ()))))
    df["n"] = n
    g = df.groupby("country").n
    return {
        "empty_rate_by_country": {k: float(v) for k, v in g.apply(lambda x: (x == 0).mean()).items()},
        "mean_matches_by_country": {k: float(v) for k, v in g.mean().items()},
        "empty_rate": float((df.n == 0).mean()),
        "mean_matches": float(df.n.mean()),
    }


def save_record(record: dict) -> Path:
    rec_dir = paths.SUB_DIR / "records"
    rec_dir.mkdir(parents=True, exist_ok=True)
    p = rec_dir / f"{record['sub_id']}.json"
    p.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return p
