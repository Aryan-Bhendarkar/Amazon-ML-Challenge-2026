import itertools

import numpy as np
import pandas as pd
import pytest

from ber.decision import assign_best_s1, expected_f05_topk, threshold_matches
from ber.metric import f05_entity


def brute_expected(p, k):
    order = np.argsort(-p)
    pred = set(order[:k].tolist())
    ev = 0.0
    for lab in itertools.product([0, 1], repeat=len(p)):
        pr = np.prod([pi if l else 1 - pi for pi, l in zip(p, lab)])
        truth = {i for i, l in enumerate(lab) if l}
        ev += pr * f05_entity(pred, truth)
    return ev


@pytest.mark.parametrize("p", [[0.9, 0.6, 0.2], [0.3, 0.2], [0.95, 0.9, 0.8, 0.1, 0.05], [0.5]])
def test_expected_f05_matches_bruteforce(p):
    p = np.array(p)
    _, ev = expected_f05_topk(p)
    for k in range(len(p) + 1):
        assert ev[k] == pytest.approx(brute_expected(p, k), abs=1e-9)


def test_assign_best_s1_keeps_one_owner():
    df = pd.DataFrame({"s1_id": ["a", "b", "a"], "cand_id": ["S2-1", "S2-1", "S3-2"], "prob": [0.7, 0.9, 0.8]})
    best = assign_best_s1(df)
    assert set(zip(best.s1_id, best.cand_id)) == {("b", "S2-1"), ("a", "S3-2")}
    assert best.set_index("cand_id").margin["S2-1"] == pytest.approx(0.2)
    assert threshold_matches(best, 0.85) == {"b": ["S2-1"]}
