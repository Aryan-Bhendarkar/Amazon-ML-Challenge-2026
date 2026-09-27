import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipelines"))


def test_hedge_unseen_open_set():
    from hedge_unseen import hedge
    s1 = pl.DataFrame({"entity_id": ["S1-a", "S1-b", "S1-c"], "country": ["US", "Narnia", "India"]})
    xm = pl.DataFrame({"s1_id": ["S1-a", "S1-b", "S1-c"], "match_id": ["S2-1", "S2-2", "S3-3"]})
    bm = pl.DataFrame({"s1_id": ["S1-a", "S1-b"], "match_id": ["S2-9", "S2-8"]})
    out, info = hedge(xm, bm, s1, seen={"US", "India"})
    got = dict(zip(out["s1_id"], out["match_id"]))
    assert got == {"S1-a": "S2-1", "S1-c": "S3-3", "S1-b": "S2-8"}      # unseen country -> base model rows
    assert info["unseen_countries"] == ["Narnia"]
