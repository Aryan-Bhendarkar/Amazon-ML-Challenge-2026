import pytest

from ber.metric import f05_entity, macro_f05, report


def test_spec_example():
    # problem statement: pred 3 (2 correct), truth 2 -> 0.714
    assert f05_entity({"S2-47", "S2-193", "S3-812"}, {"S2-47", "S3-812"}) == pytest.approx(0.7143, abs=1e-4)


def test_singletons_and_empties():
    assert f05_entity(set(), set()) == 1.0
    assert f05_entity({"S2-1"}, set()) == 0.0
    assert f05_entity(set(), {"S2-1"}) == 0.0
    assert f05_entity({"S2-9"}, {"S2-1"}) == 0.0


def test_macro_counts_missing_pred_as_empty():
    truth = {"a": set(), "b": {"S2-1"}}
    assert macro_f05({}, truth) == pytest.approx(0.5)
    assert macro_f05({"b": ["S2-1"]}, truth) == 1.0


def test_report_keys():
    r = report({"b": ["S2-1", "S3-2"]}, {"a": set(), "b": {"S2-1"}}, {"a": "US", "b": "India"})
    assert r["f05_by_country"]["US"] == 1.0
    assert 0 < r["f05_macro"] < 1


def test_paired_bootstrap_detects_improvement():
    import pandas as pd
    from ber.metric import paired_bootstrap
    a = pd.DataFrame({"s1_id": list("abcdefghij"), "f05": [0.5] * 10})
    b = pd.DataFrame({"s1_id": list("abcdefghij"), "f05": [0.6] * 10})
    r = paired_bootstrap(a, b)
    assert r["delta"] == pytest.approx(0.1) and r["p_not_better"] == 0.0
