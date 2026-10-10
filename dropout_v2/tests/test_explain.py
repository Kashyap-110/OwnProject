from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from dropout_v2 import config, explain, modeling
from dropout_v2.tests.test_modeling import PARAMS, toy_data


class TestReasons(unittest.TestCase):
    def test_positive_contributions_in_descending_order(self):
        values = pd.DataFrame({
            "cur_att_rate": [0.4, 0.95],
            "prev_status": pd.Categorical(["NOT_IN_PREV_YEAR", "PROMOTED"]),
            "over_age": [2.0, 0.0],
        })
        contrib = np.array([[0.5, 0.2, -0.1],
                            [-1.0, 0.3, 0.6]])
        out = explain.top_reasons(contrib, values, k=3)
        self.assertEqual(out.loc[0, "reason_1"], "Attendance this year so far: 40%")
        self.assertEqual(out.loc[0, "reason_2"], "Last year: not in school records last year")
        self.assertEqual(out.loc[0, "reason_3"], "")
        self.assertEqual(out.loc[1, "reason_1"], "Age normal for the class")
        self.assertEqual(out.loc[1, "reason_2"], "Last year: promoted last year")

    def test_sensitive_features_never_cited(self):
        values = pd.DataFrame({
            "gender": pd.Categorical(["FEMALE"]),
            "social_category": pd.Categorical(["SC"]),
            "district": pd.Categorical(["KURNOOL"]),
            "prev_att_rate": [0.55],
        })
        contrib = np.array([[0.9, 0.8, 0.7, 0.1]])
        out = explain.top_reasons(contrib, values, k=3)
        self.assertEqual(out.loc[0, "reason_1"], "Attendance last year: 55%")
        self.assertEqual(out.loc[0, "reason_2"], "")

    def test_missing_value_text(self):
        values = pd.DataFrame({"home_school_km": [np.nan], "cur_last_month_rate": [np.nan]})
        out = explain.top_reasons(np.array([[0.4, 0.9]]), values, k=2)
        self.assertEqual(out.loc[0, "reason_1"], "No attendance record for the latest month")
        self.assertEqual(out.loc[0, "reason_2"], "Distance from home to school: not recorded")

    def test_one_reason_per_topic(self):
        values = pd.DataFrame({
            "cur_att_rate": [0.3], "cur_last_month_rate": [0.1], "cur_trend": [-0.5],
            "prev_att_rate": [0.6], "over_age": [2.0], "age_t": [14.0],
        })
        contrib = np.array([[0.9, 0.8, 0.7, 0.3, 0.2, 0.25]])
        out = explain.top_reasons(contrib, values, k=3)
        self.assertEqual(out.loc[0, "reason_1"], "Attendance this year so far: 30%")
        self.assertEqual(out.loc[0, "reason_2"], "Attendance last year: 60%")
        self.assertEqual(out.loc[0, "reason_3"], "Age 14")

    def test_data_availability_features_never_cited(self):
        values = pd.DataFrame({"school_prev_n": [1390.0], "cur_working_days": [44.0], "prev_att_rate": [0.5]})
        out = explain.top_reasons(np.array([[0.9, 0.8, 0.1]]), values, k=2)
        self.assertEqual(out.loc[0, "reason_1"], "Attendance last year: 50%")
        self.assertEqual(out.loc[0, "reason_2"], "")


class TestContributions(unittest.TestCase):
    def test_shapes_and_importance(self):
        df, y = toy_data(n=3000, seed=4)
        cats = modeling.fit_categories(df, ["group"], min_count=10)
        X = modeling.prepare_matrix(df, ["x1", "x2"], ["group"], cats)
        booster = modeling.train_fixed(X, y, np.ones(len(y)), PARAMS, n_rounds=300)
        contrib = explain.contributions(booster, X, num_iteration=300, chunk=1000)
        self.assertEqual(contrib.shape, (3000, 3))
        raw = booster.predict(X, num_iteration=300, raw_score=True)
        bias = booster.predict(X.iloc[:1], num_iteration=300, pred_contrib=True)[0, -1]
        np.testing.assert_allclose(contrib.sum(axis=1) + bias, raw, atol=1e-4)
        imp = explain.global_importance(booster, X, num_iteration=300)
        self.assertEqual(imp.iloc[0]["feature"], "x1")


if __name__ == "__main__":
    unittest.main()
