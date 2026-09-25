import numpy as np
import pandas as pd

from ber import setdecision as sd
from ber.metric import f05_entity


def _assigned():
    return pd.DataFrame({"s1_id": ["a", "a", "a", "b", "c"], "cand_id": ["S2-1", "S3-2", "S2-3", "S2-4", "S3-5"],
                         "prob": [0.9, 0.8, 0.3, 0.6, 0.95]})


def test_prefix_f_matches_metric():
    lab = np.array([True, False, True])
    f = sd.prefix_f(lab, n_true=3, kmax=4)
    truth = {"x", "z", "w"}
    names = ["x", "y", "z"]
    for k in range(4):
        assert abs(f[k] - f05_entity(set(names[:k]), truth)) < 1e-12
    assert sd.prefix_f(np.array([False]), 0, 3)[0] == 1.0


def test_oracle_and_topk():
    truth = {"a": {"S2-1", "S3-2"}, "b": set(), "c": {"S3-5", "S2-9"}, "d": set()}
    o = sd.oracle_k(_assigned(), truth)
    assert o.loc["a", "k_star"] == 2 and o.loc["b", "k_star"] == 0 and o.loc["c", "k_star"] == 1
    assert o.loc["d", "k_star"] == 0 and o.loc["d", "f_star"] == 1.0
    m = sd.topk_matches(_assigned(), o["k_star"])
    assert m == {"a": ["S2-1", "S3-2"], "c": ["S3-5"]}


def test_choose_k_one_hot_is_identity_and_hedges():
    C = 6
    P = np.eye(C)
    assert (sd.choose_k(P) == np.arange(C)).all()
    # 50/50 between c=0 (singleton) and c=3: predicting 3 is worth 0.5*1.0, empty 0.5*1.0 -> tie -> smallest
    p = np.zeros((1, C)); p[0, 0] = 0.4; p[0, 3] = 0.6
    assert sd.choose_k(p)[0] == 3


def test_profile_and_rel():
    prof = sd.s1_profile(_assigned(), ["a", "b", "c", "d"], k=3)
    assert prof.loc["a", "p1"] == np.float32(0.9) and prof.loc["d", "n_assigned"] == 0
    assert abs(prof.loc["a", "gap1"] - 0.1) < 1e-6 and prof.loc["a", "s3_ge50"] == 1
    r = sd.rel_features(_assigned()).set_index("cand_id")
    assert r.loc["S3-2", "r_rank"] == 1 and abs(r.loc["S2-3", "r_ratio"] - 0.3 / 0.9) < 1e-6
