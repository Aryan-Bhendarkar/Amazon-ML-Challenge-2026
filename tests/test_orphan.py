import numpy as np
import polars as pl

from ber import orphan


def _s1(n=4000):
    rng = np.random.default_rng(0)
    names = [f"name{i % 3000}" for i in range(n)]          # 2000 S1 share a name, 2000 unique
    return pl.DataFrame({"entity_id": [f"S1-{i}" for i in range(n)], "country": ["US"] * n, "n_core": names,
                         "a_full": [f"{i} main st" for i in range(n)], "a_street": ["main st"] * n,
                         "a_house": [str(i) for i in range(n)]}).sample(fraction=1.0, seed=int(rng.integers(9)))


def test_mask_deterministic_and_rate():
    s1 = _s1()
    a = orphan.removal_mask(s1, 0.19, "uniform")
    b = orphan.removal_mask(s1.reverse(), 0.19, "uniform")
    ra = dict(zip(a["entity_id"], a["removed"]))
    assert all(ra[k] == v for k, v in zip(b["entity_id"], b["removed"]))
    assert abs(a["removed"].mean() - 0.19) < 0.03
    assert orphan.removal_mask(s1, 0.0)["removed"].sum() == 0


def test_biased_mask_prefers_shared():
    s1 = _s1()
    m = orphan.removal_mask(s1, 0.19, "biased")
    sh = orphan.shared_flag(s1)
    rem = m["removed"].to_numpy()
    assert abs(rem.mean() - 0.19) < 0.03
    assert rem[sh].mean() > 2 * rem[~sh].mean()


def test_loss_decomposition_sums():
    truth = {"a": set(), "b": {"x", "y"}, "c": {"z"}, "d": {"w"}}
    pred = {"a": ["q"], "b": ["x", "bad"], "d": ["w"]}
    d = orphan.loss_decomposition(pred, truth)
    from ber.metric import macro_f05
    assert abs(d["loss_total"] - (1 - macro_f05(pred, truth))) < 1e-12
    assert d["loss_fp_singleton"] == 0.25


def test_derived_lfrac_roundtrip():
    from ber import ctx_features as cf
    X = pl.DataFrame({"ex_lfrac_cmax": [-3.26, -3.24, None], "other": [1, 2, 3]})
    feats = ["other", "ex_lfrac_cmax_q05"]
    assert cf.expand_derived(feats) == ["other", "ex_lfrac_cmax"]
    Y = cf.add_derived(X, feats)
    assert Y["ex_lfrac_cmax_q05"].to_list()[:2] == [-3.5, -3.0] and Y["ex_lfrac_cmax_q05"][2] is None
