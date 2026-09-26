import numpy as np
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


def test_v2_rarity_is_density_invariant():
    s1 = pl.DataFrame({"country": ["X"] * 4, "n_core": ["solo co", "acme labs", "acme co", "zeta co"],
                       "a_full": ["9 oak rd"] * 4, "a_street": ["oak rd"] * 4, "a_house": ["9"] * 4})
    pool = pl.DataFrame({"country": ["X"], "n_core": ["solo co"], "a_full": ["9 oak rd"]})
    ctx = cf.SplitContext.from_frames(s1, pool, version=2)
    F = cf.add_features(_pairs(), ctx, groups=("G3",), workers=1).sort("cand_id")
    r = {c: F[c].to_list() for c in F.columns}
    # extras vs S1 'solo co': r0 none, r1 'holdings' (unseen), r2 'cp' (unseen), r3 'acme'(df 2) + 'labs'(df 1)
    assert "ex_idf_max" not in r and r["ex_unseen_n"] == [0, 1, 1, 0]
    assert abs(r["ex_ldf_max"][3] - np.log1p(2)) < 1e-6 and abs(r["ex_ldf_min"][3] - np.log1p(1)) < 1e-6
    assert r["ex_ldf_max"][0] == -1.0 and abs(r["mi_ldf_max"][2] - np.log1p(3)) < 1e-6   # missing "co": df 3
    assert r["ex_lfrac_cmax"][1] is None                                  # no common (df>=50) tokens here
    # doubling the S1 table (denser split) must not change rare-token features of the same tokens
    ctx2 = cf.SplitContext.from_frames(pl.concat([s1, s1.with_columns(pl.lit("Y").alias("country"))]), pool, version=2)
    F2 = cf.add_features(_pairs(), ctx2, groups=("G3",), workers=1).sort("cand_id")
    assert F2["ex_ldf_max"].to_list() == r["ex_ldf_max"]


def test_v4_name_twins():
    s1 = pl.DataFrame({"entity_id": ["c", "t1", "t2"], "country": ["X"] * 3,
                       "n_core": ["solo co", "solo co", "acme labs"],
                       "a_full": ["9 oak rd", "8 elm st", "1 main st"], "a_street": ["oak rd", "elm st", "main st"],
                       "a_house": ["9", "8", "1"]})
    pool = pl.DataFrame({"country": ["X"], "n_core": ["solo co"], "a_full": ["9 oak rd"]})
    ctx = cf.SplitContext.from_frames(s1, pool, version=4)
    P = _pairs().with_columns(pl.Series("a_full_c", ["9 oak rd", "8 elm st", "", "1 main st"]),
                              pl.Series("a_street_c", ["oak rd", "elm st", "", "main st"]),
                              pl.Series("a_house_c", ["9", "8", "", "1"]))
    F = cf.add_features(P, ctx, groups=("G1", "G6"), workers=1).sort("cand_id")
    r = {c: F[c].to_list() for c in F.columns}
    # r0 'co solo' (key of c and t1): twin t1 at 8 elm st; own address matches -> positive margin
    assert r["twin_n"][0] == 1 and r["twin_addr_margin"][0] > 0 and r["twin_house_eq"][0] == 0
    # r1 'solo co holdings' has no twins; r3 'acme labs' twin t2 sits exactly at the candidate's address
    assert r["twin_n"][1] == 0 and r["twin_addr_margin"][1] is None
    assert r["twin_n"][3] == 1 and r["twin_addr_margin"][3] < 0 and r["twin_house_eq"][3] == 1
    assert r["twin_generic"] == [0, 0, 0, 0]


def test_v4_templated_twin_flag():
    # S1 'c' at 8 rue general dampierre; twin 't1' (same name) at 8 rue de vassy; candidate at 8 r de vassy
    s1 = pl.DataFrame({"entity_id": ["c", "t1"], "country": ["F"] * 2, "n_core": ["comite", "comite"],
                       "a_full": ["8 rue general dampierre", "8 rue de vassy"],
                       "a_street": ["rue general dampierre", "rue de vassy"], "a_house": ["8", "8"]})
    pool = pl.DataFrame({"country": ["F"], "n_core": ["comite"], "a_full": ["8 rue de vassy"]})
    ctx = cf.SplitContext.from_frames(s1, pool, version=4)
    P = pl.DataFrame({"s1_id": ["c"], "cand_id": ["r"], "country": ["F"], "n_core": ["comite"],
                      "a_full": ["8 rue general dampierre"], "a_street": ["rue general dampierre"], "a_house": ["8"],
                      "a_numbers": ["8"], "a_postcode": [""], "n_core_c": ["comite"], "a_full_c": ["8 rue de vassy"],
                      "a_street_c": ["rue de vassy"], "a_house_c": ["8"], "a_numbers_c": ["8"], "a_postcode_c": [""],
                      "a_empty_c": [False], "name_tset": [100.0]})
    F = cf.add_features(P, ctx, groups=("G6",), workers=1)
    assert F["twin_n"][0] == 1 and F["twin_house_eq"][0] == 1 and F["twin_hs_better"][0] == 1
    assert F["twin_street_ratio_margin"][0] < 0
