import polars as pl

from ber import blocking as B


def test_skeleton_collapses_transliterations():
    s = B.skeleton(pl.Series(["blu impeks praivet", "blue impex private", "pharst lksmi", "first laxmi"])).to_list()
    assert s[0] == s[1] == "bl|mpks"
    assert s[2] == s[3]


def test_hskey_and_akey():
    df = pl.DataFrame({"a_house": ["08", "8", ""], "a_street": ["greenwood rd", "rd greenwood", "greenwood rd"],
                       "a_full": ["08 greenwood rd x", "x rd greenwood 08", ""]})
    hs = df.select(B.hskey_expr()).to_series().to_list()
    assert hs[0] == hs[1] != "" and hs[2] == ""
    ak = df.select(B.akey_expr()).to_series().to_list()
    assert ak[0] == ak[1] and ak[2] == ""


def test_key_join_caps_blocks():
    q = pl.Series(["a", "b", ""])
    p = pl.Series(["a", "a", "b", "b", "b", ""])
    got = B.key_join(q, p, cap=2).sort("i", "j").rows()
    assert got == [(0, 0), (0, 1)]          # block "b" has 3 > cap records; '' never joins


def test_union_ors_bits():
    u = B.union({"tf_name": pl.DataFrame({"i": [0, 1], "j": [5, 6]}),
                 "akey": pl.DataFrame({"i": [0], "j": [5]})}).sort("i").rows()
    assert u == [(0, 5, B.RBITS["tf_name"] | B.RBITS["akey"]), (1, 6, B.RBITS["tf_name"])]


def test_nkey_num():
    q = pl.DataFrame({"n_core": ["ram marketing", "x"], "a_numbers": ["570/13", ""]})
    p = pl.DataFrame({"n_core": ["marketing ram", "ram marketing", "ram marketing ram"], "a_numbers": ["013 9", "571", "570"]})
    got = B.multikey_join(B.nkey_num_keys(q, "i"), B.nkey_num_keys(p, "j"), cap=50).sort("i", "j").rows()
    assert got == [(0, 0), (0, 2)]      # shared 13 (zero-stripped) / 570 via unique tokens; 571 differs; no number -> none
