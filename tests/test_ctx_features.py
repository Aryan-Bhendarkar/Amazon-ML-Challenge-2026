import polars as pl

from ber import ctx_features as cf


def _ctx():
    s1 = pl.DataFrame({
        "country": ["X", "X", "X", "Y"],
        "n_core": ["acme labs", "labs acme", "solo co", "acme labs"],
        "a_full": ["1 main st", "2 elm st", "9 oak rd", "1 main st"],
        "a_street": ["main st", "elm st", "oak rd", "main st"],
        "a_house": ["1", "2", "9", "1"],
    }).with_columns(cf.key_exprs())
    cnt = lambda df, k: df.filter(pl.col(k) != "").group_by("country", k).len("cnt")  # noqa: E731
    idf = pl.DataFrame({"country": ["X", "X", "X", "X"], "tok": ["acme", "labs", "solo", "co"],
                        "idf": [0.4, 0.4, 1.1, 1.1]}, schema_overrides={"idf": pl.Float32})
    pool = pl.DataFrame({"country": ["X", "X"], "n_key": ["co solo", "co solo"], "a_key": ["9 oak rd", ""]})
    return cf.SplitContext(cnt(s1, "n_key"), cnt(s1, "a_key"), cnt(s1, "hs_key"),
                           cnt(pool, "n_key"), cnt(pool, "a_key"), idf)


def _pairs():
    n = 4
    return pl.DataFrame({
        "s1_id": ["c"] * n, "cand_id": [f"r{i}" for i in range(n)], "country": ["X"] * n,
        "n_core": ["solo co"] * n, "a_full": ["9 oak rd"] * n, "a_street": ["oak rd"] * n,
        "a_house": ["9"] * n, "a_numbers": ["9 4"] * n, "a_postcode": [""] * n,
        "n_core_c": ["co solo", "solo co holdings", "solo cp", "acme labs"],
        "a_full_c": ["9 oak rd", "10 oak rd", "", "9 oak rd"],
        "a_street_c": ["oak rd", "oak rd", "", "oak rd"],
        "a_house_c": ["9", "10", "", "9"], "a_numbers_c": ["9 4", "10 5", "", "9"],
        "a_postcode_c": ["", "", "", ""], "a_empty_c": [False, False, True, False],
        "name_tset": [100.0, 100.0, 85.0, 20.0],
    })


def test_features_end_to_end():
    F = cf.add_features(_pairs(), _ctx(), workers=1).sort("cand_id")
    r = {c: F[c].to_list() for c in F.columns}
    # G1: cand name key "co solo" -> 1 S1 in X; "acme labs" -> 2 S1 (order-free key); pool counts
    assert r["s1_name_c"] == [1, 0, 0, 2] and r["s1_name_self"] == [1] * 4
    assert r["n_key_equal"] == [1, 0, 0, 0] and r["pool_name_c"][0] == 2
    # G2
    assert r["a_key_equal"] == [1, 0, 0, 1] and r["s1_addr_c"] == [1, 0, 0, 1]
    assert r["hs_key_equal"] == [1, 0, 0, 1]
    # G3: r1 is a pure insertion of a business word with the S1 as prefix; r2 a substitution typo
    assert r["edit_type"] == [0, 1, 3, 3]
    assert r["ex_biz"][1] == 1 and r["s1_prefix_of_c"] == [0, 1, 0, 0]
    assert r["sub_jw"][2] > 0.6 and r["sub_jw"][0] == -1.0
    assert r["ex_idf_max"][1] > 1.1                        # unseen token -> country max idf
    # G4
    assert r["house_lev"] == [0, 2, -1, 0] and r["house_same_len"][1] == 0
    assert abs(r["house_logdiff"][1] - 0.6931) < 1e-3
    assert r["sec_num_rel"] == [1, 2, 0, 0]
    # G5: siblings = name_tset>=80 with a house: r0, r1. For r0 the other sib (r1) has house 10.
    assert r["sib_n"] == [1, 1, 2, 2]
    assert r["sib_house_agree"] == [0, 0, -1, 1]
    assert r["sib_s1house_agree"] == [0, 1, 1, 1]
