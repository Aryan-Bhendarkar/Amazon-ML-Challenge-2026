import numpy as np
import pandas as pd

from ber import dmval
from ber.metric import f05_entity


def _frame():
    a = pd.DataFrame({"s1_id": ["a", "a", "b", "c", "c"], "cand_id": list("vwxyz"),
                      "prob": [0.99, 0.8, 0.85, 0.9, 0.3], "lab": [True, False, False, True, True]})
    n_true = pd.Series({"a": 1, "b": 0, "c": 3, "d": 0})
    return a, n_true


def test_clean_matches_official_metric():
    a, n_true = _frame()
    f = dmval.entity_f05(a, n_true, 0.75)
    truth = {"a": {"v"}, "b": set(), "c": {"y", "z", "q"}, "d": set()}
    pred = {"a": ["v", "w"], "b": ["x"], "c": ["y"]}
    for k in truth:
        assert abs(f[k] - f05_entity(set(pred.get(k, [])), truth[k])) < 1e-12


def test_weight_increases_fp_only_in_band():
    a, n_true = _frame()
    c = dmval.copies(a, 3.0, 0.2, 0.975, seed=0)
    assert c.tolist() == [1, 3, 3, 1, 1]           # 0.99 out of band; positives never replicated
    f = dmval.entity_f05(a, n_true, 0.75, c)
    assert f["a"] < dmval.entity_f05(a, n_true, 0.75)["a"] and f["b"] == 0 and f["d"] == 1


def test_negative_weight():
    assert abs(dmval.negative_weight(2.0, 0.2, 0.1) - 3.0) < 1e-12   # (0.4 - 0.1) / 0.1
    assert dmval.negative_weight(0.5, 0.2, 0.1) == 1.0
