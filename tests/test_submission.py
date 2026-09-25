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
