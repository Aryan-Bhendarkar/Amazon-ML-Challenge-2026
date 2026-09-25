import pandas as pd

from ber.split import eval_ids


def test_fold0x_is_fold0_minus_mini():
    f = pd.DataFrame({"s1_id": list("abcde"), "fold": [0, 0, 0, 1, 1], "mini": [True, False, False, False, True]})
    assert eval_ids(f, "fold0x") == {"b", "c"}
    assert eval_ids(f, "fold0x").isdisjoint(eval_ids(f, "mini"))
