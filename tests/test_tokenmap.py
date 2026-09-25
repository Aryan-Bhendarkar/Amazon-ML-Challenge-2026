import polars as pl

from ber import tokenmap
from ber.normalize import normalize_name


def test_learn_positional_majority_and_filters():
    c = pl.Series(["gret phaumdesn", "gret phaumdesn", "gret phaumdesn", "gret foo", "pvt ltd x"])
    s = pl.Series(["great foundation", "great foundation", "great foundation", "great bar", "pvt ltd"])
    tmap, st = tokenmap.learn(tokenmap.aligned_pairs(c, s), min_support=3, min_precision=0.5)
    assert tmap == {"gret": "great", "phaumdesn": "foundation"}      # 'foo' support 1; unequal length skipped
    assert st["n_map"] == 2


def test_map_applies_only_to_non_latin_names():
    tmap = {"siv": "shiv", "prodyusr": "producer"}
    assert normalize_name("Siv Prodyusr", token_map=tmap).core == "siv prodyusr"          # Latin: untouched
    native = normalize_name("शिव प्रोड्यूसर", token_map=None).core
    mapped = normalize_name("शिव प्रोड्यूसर", token_map={t: "x" + t for t in native.split()}).core
    assert mapped == " ".join("x" + t for t in native.split())


def test_mapped_legal_tokens_become_legal_forms():
    n = normalize_name("गुरु प्राइवेट", token_map={"praaivett": "pvt", "prin": "pvt", "praaiveett": "pvt",
                                                   **{t: "pvt" for t in normalize_name("प्राइवेट").full.split()}})
    assert n.legal == "pvt" and "pvt" not in n.core.split()
