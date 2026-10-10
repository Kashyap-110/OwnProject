from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from dropout_v2 import evaluate

Y = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 1], dtype=float)
P = np.array([.9, .8, .7, .6, .5, .4, .3, .2, .1, .05])


class TestMetrics(unittest.TestCase):
    def test_topk_precision_recall_lift(self):
        table = evaluate.topk_table(Y, P, [0.2, 0.3]).set_index("top_k")
        self.assertEqual(int(table.loc[0.2, "n_flagged"]), 2)
        self.assertAlmostEqual(table.loc[0.2, "precision"], 0.5)
        self.assertAlmostEqual(table.loc[0.2, "recall"], 1 / 3)
        self.assertAlmostEqual(table.loc[0.3, "precision"], 2 / 3)
        self.assertAlmostEqual(table.loc[0.3, "lift"], (2 / 3) / 0.3)

    def test_core_metrics(self):
        m = evaluate.core_metrics(Y, P)
        self.assertEqual(m["n"], 10)
        self.assertEqual(m["positives"], 3)
        self.assertTrue(0 <= m["auc"] <= 1)
        single = evaluate.core_metrics(np.zeros(5), np.linspace(0, 1, 5))
        self.assertTrue(np.isnan(single["auc"]))
        self.assertTrue(np.isnan(single["ap"]))

    def test_breakdown_handles_single_class_group(self):
        df = pd.DataFrame({"y": Y, "p": P, "g": ["a"] * 5 + ["b"] * 5})
        df.loc[df["g"] == "b", "y"] = 0.0
        out = evaluate.breakdown(df, "y", "p", "g", k=0.2).set_index("g")
        self.assertTrue(np.isnan(out.loc["b", "auc"]))
        self.assertEqual(int(out.loc["a", "n"]), 5)
        self.assertAlmostEqual(out.loc["a", "recall_at_top_k"], 0.5)   # positives 0.9 and 0.7; threshold keeps 0.9, 0.8

    def test_calibration_table(self):
        rng = np.random.default_rng(0)
        p = rng.random(5000)
        y = (rng.random(5000) < p).astype(float)
        table = evaluate.calibration_table(y, p, bins=10)
        self.assertEqual(len(table), 10)
        self.assertLess((table["mean_predicted"] - table["observed_rate"]).abs().max(), 0.06)

    def test_bootstrap_interval_contains_point(self):
        rng = np.random.default_rng(1)
        p = rng.random(3000)
        y = (rng.random(3000) < p * 0.3).astype(float)
        ci = evaluate.bootstrap_ci(y, p, reps=30, max_rows=2000, seed=0)
        point = evaluate.core_metrics(y, p)
        self.assertLessEqual(ci["auc"][0], point["auc"] + 0.02)
        self.assertGreaterEqual(ci["auc"][1], point["auc"] - 0.02)
        self.assertLess(ci["ap"][0], ci["ap"][1])

    def test_single_feature_auc(self):
        rng = np.random.default_rng(2)
        y = (rng.random(2000) < 0.2).astype(float)
        X = pd.DataFrame({"leak": y + rng.normal(0, 0.01, 2000), "noise": rng.normal(size=2000)})
        out = evaluate.single_feature_auc(X, y).set_index("feature")
        self.assertGreater(out.loc["leak", "auc"], 0.99)
        self.assertLess(out.loc["noise", "auc"], 0.6)

    def test_rule_baseline(self):
        df = pd.DataFrame({"cur_att_rate": [0.9, np.nan], "prev_att_rate": [0.5, 0.8]})
        np.testing.assert_allclose(evaluate.rule_baseline(df, "jul"), [0.1, 1.0])
        np.testing.assert_allclose(evaluate.rule_baseline(df, "start"), [0.5, 0.2])

    def test_logistic_baseline(self):
        rng = np.random.default_rng(3)
        n = 3000
        x = rng.normal(size=n)
        g = pd.Categorical(rng.choice(["A", "B"], size=n))
        y = (rng.random(n) < 1 / (1 + np.exp(-(2 * x - 1)))).astype(float)
        df = pd.DataFrame({"x": x, "g": g})
        scores = evaluate.logistic_baseline(df.iloc[:2000], y[:2000], df.iloc[2000:], ["x"], ["g"], sample_rows=1500)
        self.assertEqual(len(scores), 1000)
        self.assertGreater(evaluate.core_metrics(y[2000:], scores)["auc"], 0.75)

    def test_plots_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            pr = evaluate.plot_pr_curve(Y, {"model": P, "rule": P[::-1]}, Path(tmp) / "pr.png", "test")
            cal = evaluate.plot_calibration(evaluate.calibration_table(Y, P, bins=3), Path(tmp) / "cal.png", "test")
            imp = evaluate.plot_importance(pd.DataFrame({"feature": ["a", "b"], "mean_abs_contribution": [2.0, 1.0]}),
                                           Path(tmp) / "imp.png", "test")
            for path in (pr, cal, imp):
                self.assertTrue(path.exists() and path.stat().st_size > 0)


if __name__ == "__main__":
    unittest.main()
