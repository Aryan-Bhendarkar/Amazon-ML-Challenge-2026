"""Seed bags: a run's artifacts either hold one `model.lgb` or a `bag.json` ({"models": [files]}) whose LightGBM
members are averaged in probability space (lead 08:05: seed-bag = variance reduction). Every consumer loads models through
`load_model(art_dir)` so single models and bags are interchangeable (dm_val, predict_test_v1)."""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np


class Bag:
    def __init__(self, boosters: list[lgb.Booster]):
        self.boosters = boosters

    def predict(self, X, num_threads: int | None = None) -> np.ndarray:
        kw = {} if num_threads is None else {"num_threads": num_threads}
        return np.mean([b.predict(X, **kw) for b in self.boosters], axis=0)


def load_model(art: Path):
    bj = Path(art) / "bag.json"
    if bj.exists():
        return Bag([lgb.Booster(model_file=str(Path(art) / f)) for f in json.loads(bj.read_text())["models"]])
    return lgb.Booster(model_file=str(Path(art) / "model.lgb"))
