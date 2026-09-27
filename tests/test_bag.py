import json

import lightgbm as lgb
import numpy as np

from ber import bag


def test_bag_averages_members(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.random((500, 4)); y = (X[:, 0] + 0.1 * rng.random(500) > 0.5).astype(int)
    files = []
    for s in (1, 2):
        m = lgb.train(dict(objective="binary", verbose=-1, seed=s, bagging_fraction=0.7, bagging_freq=1), lgb.Dataset(X, y), 20)
        f = f"model_s{s}.lgb"; m.save_model(str(tmp_path / f)); files.append(f)
    (tmp_path / "bag.json").write_text(json.dumps({"models": files}))
    b = bag.load_model(tmp_path)
    ref = np.mean([lgb.Booster(model_file=str(tmp_path / f)).predict(X) for f in files], axis=0)
    assert np.allclose(b.predict(X), ref)
    (tmp_path / "bag.json").unlink(); m.save_model(str(tmp_path / "model.lgb"))
    assert isinstance(bag.load_model(tmp_path), lgb.Booster)
