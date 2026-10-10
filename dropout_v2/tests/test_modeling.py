from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from dropout_v2 import config, modeling


def toy_data(n=6000, seed=0):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    group = rng.choice(["A", "B", "C"], size=n, p=[0.5, 0.45, 0.05])
    logit = -2.5 + 1.8 * x1 + np.where(group == "B", 0.8, 0.0)
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(float)
    df = pd.DataFrame({"x1": x1, "x2": rng.normal(size=n), "group": pd.Categorical(group)})
    return df, y


PARAMS = dict(config.LGB_PARAMS, num_leaves=15, min_data_in_leaf=20, num_threads=2)


class TestCategories(unittest.TestCase):
    def test_unseen_category_maps_to_other(self):
        train = pd.DataFrame({"g": pd.Categorical(["A"] * 100 + ["B"] * 100 + ["C"] * 5)})
        cats = modeling.fit_categories(train, ["g"], min_count=10)
        self.assertEqual(cats["g"], ["A", "B", config.OTHER, config.UNKNOWN])
        test = pd.DataFrame({"g": pd.Categorical(["A", "Z", None, "C", "B"])})
        X = modeling.prepare_matrix(test, [], ["g"], cats)
        self.assertEqual(list(X["g"].astype(str)), ["A", config.OTHER, config.UNKNOWN, config.OTHER, "B"])
        self.assertEqual(list(X["g"].cat.categories), cats["g"])

    def test_object_column_and_numeric_cast(self):
        df = pd.DataFrame({"g": ["A", "B", None], "x": [1, None, 3]})
        cats = modeling.fit_categories(df, ["g"], min_count=1)
        X = modeling.prepare_matrix(df, ["x"], ["g"], cats)
        self.assertEqual(X["x"].dtype, np.float32)
        self.assertEqual(list(X["g"].astype(str)), ["A", "B", config.UNKNOWN])


class TestNegativeSampling(unittest.TestCase):
    def test_keeps_positives_deterministic_and_reweights(self):
        keys = np.arange(100000, dtype="int64") * 7919
        y = np.zeros(100000)
        y[:1000] = 1
        mask, weights, rate = modeling.negative_sample(keys, y, neg_per_pos=5.0, rate_min=0.05)
        self.assertTrue(mask[:1000].all())
        mask2, _, _ = modeling.negative_sample(keys, y, neg_per_pos=5.0, rate_min=0.05)
        self.assertTrue((mask == mask2).all())
        self.assertAlmostEqual(rate, 0.0505, places=3)
        restored = weights[mask & (y == 0)].sum()
        self.assertAlmostEqual(restored / 99000, 1.0, delta=0.05)

    def test_no_sampling_when_balanced(self):
        keys = np.arange(100, dtype="int64")
        y = np.array([1.0, 0.0] * 50)
        mask, weights, rate = modeling.negative_sample(keys, y, neg_per_pos=5.0, rate_min=0.05)
        self.assertEqual(rate, 1.0)
        self.assertTrue(mask.all())
        self.assertTrue((weights == 1.0).all())


class TestTraining(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        df, y = toy_data()
        cls.cats = modeling.fit_categories(df.iloc[:4000], ["group"], min_count=10)
        cls.Xtr = modeling.prepare_matrix(df.iloc[:4000], ["x1", "x2"], ["group"], cls.cats)
        cls.Xva = modeling.prepare_matrix(df.iloc[4000:], ["x1", "x2"], ["group"], cls.cats)
        cls.ytr, cls.yva = y[:4000], y[4000:]
        cls.booster, cls.best = modeling.train_with_early_stopping(
            cls.Xtr, cls.ytr, np.ones(4000), cls.Xva, cls.yva, PARAMS, max_rounds=300, patience=20)

    def test_early_stopping_returns_best_iteration(self):
        self.assertGreaterEqual(self.best, 1)
        self.assertLessEqual(self.best, 300)

    def test_calibrator_bounded_and_monotonic(self):
        raw = self.booster.predict(self.Xva, num_iteration=self.best)
        cal = modeling.fit_calibrator(raw, self.yva)
        order = np.argsort(raw)
        calibrated = cal.predict(raw[order])
        self.assertTrue(((calibrated >= 0) & (calibrated <= 1)).all())
        self.assertTrue((np.diff(calibrated) >= -1e-12).all())

    def test_bundle_round_trip(self):
        raw = self.booster.predict(self.Xva, num_iteration=self.best)
        bundle = modeling.ModelBundle(
            snapshot="jul", numeric=["x1", "x2"], categorical=["group"], categories=self.cats,
            booster=self.booster, best_iteration=self.best, calibrator=modeling.fit_calibrator(raw, self.yva),
            neg_rate=1.0, train_years=["2023-24"], calibration_year="2024-25", params=PARAMS)
        df, _ = toy_data(n=300, seed=9)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bundle_jul.joblib"
            modeling.save_bundle(bundle, path)
            loaded = modeling.load_bundle(path)
            np.testing.assert_allclose(bundle.predict(df), loaded.predict(df))
            self.assertTrue((Path(tmp) / "model_jul.txt").exists())

    def test_train_fixed_runs_exact_rounds(self):
        booster = modeling.train_fixed(self.Xtr, self.ytr, np.ones(4000), PARAMS, n_rounds=7)
        self.assertEqual(booster.current_iteration(), 7)


if __name__ == "__main__":
    unittest.main()
