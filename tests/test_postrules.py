import pandas as pd
import polars as pl

from ber import postrules


def _frame():
    s1 = postrules.with_keys(pl.DataFrame({
        "entity_id": ["a", "b", "c", "d"], "country": ["X", "X", "X", "Y"],
        "n_core": ["acme labs", "labs acme", "solo co", "acme labs"],
        "a_full": ["1 main st", "2 elm st", "9 oak rd", "1 main st"]}))
    return s1, *postrules.s1_key_counts(s1)


def test_counts_are_per_country_and_token_order_free():
    s1, nk, ak = _frame()
    d = {(r["country"], r["n_key"]): r["s1_same_name"] for r in nk.to_dicts()}
    assert d[("X", "acme labs")] == 2 and d[("Y", "acme labs")] == 1 and d[("X", "co solo")] == 1


def test_rules():
    s1, nk, ak = _frame()
    A = pl.DataFrame({
        "s1_id": ["c", "a", "c", "c"], "cand_id": ["r1", "r2", "r3", "r4"], "prob": [0.1, 0.9, 0.1, 0.1],
        "n_core": ["solo co", "acme labs", "solo co", "solo co"],
        "n_key": ["co solo", "acme labs", "co solo", "co solo"],
        "a_key": ["9 oak rd"] * 4,
        "country_c": ["X"] * 4,
        "n_core_c": ["co solo", "acme labs", "zzz", "solo co holdings"],
        "n_key_c": ["co solo", "acme labs", "zzz", "co holdings solo"],
        "a_key_c": ["", "", "9 oak rd", "9 oak rd"],
        "a_empty_c": [True, True, False, False]})
    f = postrules.rule_flags(A, nk, ak, workers=1).to_pandas()
    assert f.ruleA.tolist() == [True, False, False, False]        # unique exact name, empty addr
    assert f.dropA_cond.tolist() == [False, True, False, False]   # name shared by 2 S1 in X
    assert f.ruleB.tolist() == [False, False, True, False]        # same addr, unrelated name; not the extra-word copy
    keep = postrules.keep_mask(f, 0.5)
    assert keep.tolist() == [True, False, True, False]
