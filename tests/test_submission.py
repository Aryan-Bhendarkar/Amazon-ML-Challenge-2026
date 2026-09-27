from ber.submission import MATCH_HEADER, write_id_lists


def test_writer_lf_tabs_dedupe(tmp_path):
    p = tmp_path / "m.tsv"
    st = write_id_lists(p, {"S1-1": ["S2-5", "S2-5", " S3-7", "S1-9"], "S1-3": []},
                        ["S1-1", "S1-2", "S1-3"], MATCH_HEADER)
    raw = p.read_bytes()
    assert b"\r" not in raw
    lines = raw.decode("utf-8").split("\n")
    assert lines[0] == "source1_entity_id\tmatched_entity_ids"
    assert lines[1] == "S1-1\tS2-5,S3-7"
    assert lines[2] == "S1-2\t" and lines[3] == "S1-3\t"
    assert st == {"rows": 3, "nonempty": 1, "ids": 2, "dropped_bad_ids": 1}


def test_integrity_and_readback(tmp_path):
    import pandas as pd
    from ber.submission import integrity_errors, readback_errors
    order = ["S1-1", "S1-2", "S1-3"]
    ok = pd.DataFrame({"s1_id": ["S1-1", "S1-1", "S1-2"], "match_id": ["S2-1", "S3-2", "S2-3"]})
    assert integrity_errors(ok, order) == []
    two_owners = pd.DataFrame({"s1_id": ["S1-1", "S1-2"], "match_id": ["S2-1", "S2-1"]})
    assert any("more than one S1" in e for e in integrity_errors(two_owners, order))
    unknown = pd.DataFrame({"s1_id": ["S1-9"], "match_id": ["S2-1"]})
    assert any("not in test_source1" in e for e in integrity_errors(unknown, order))
    assert any("duplicate S1" in e for e in integrity_errors(ok, order + ["S1-1"]))
    assert any("not S2-/S3-" in e for e in integrity_errors(pd.DataFrame({"s1_id": ["S1-1"], "match_id": ["S1-2"]}), order))
    p = tmp_path / "m.tsv"
    write_id_lists(p, {"S1-1": ["S2-1"]}, order, MATCH_HEADER)
    assert readback_errors(p, order) == []
    p.write_text("source1_entity_id\tmatched_entity_ids\nS1-1\tS2-1\nS1-1\t\n", encoding="utf-8", newline="\n")
    errs = readback_errors(p, order)
    assert any("duplicate" in e for e in errs) and any("missing" in e for e in errs)
