from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from dropout_v2 import attendance

NAN = np.nan


def feats(att_rows, wd_rows, months, prefix="cur_"):
    att = np.array(att_rows, dtype="float64")
    wd = np.array(wd_rows, dtype="float64")
    return attendance.attendance_features(att, wd, months, prefix)


class TestAttendanceFeatures(unittest.TestCase):
    def test_rate_is_pooled_not_mean_of_months(self):
        f = feats([[10, 20]], [[10, 40]], ["jun", "jul"])
        self.assertAlmostEqual(f["cur_att_rate"][0], 30 / 50)
        self.assertEqual(f["cur_days_attended"][0], 30)
        self.assertEqual(f["cur_working_days"][0], 50)

    def test_trailing_zero_months_and_last_month(self):
        f = feats([[20, 15, 0, 0]], [[22, 22, 22, 22]], ["jun", "jul", "aug", "sep"])
        self.assertEqual(f["cur_trailing_zero_months"][0], 2)
        self.assertEqual(f["cur_last_att_month"][0], 2)
        self.assertEqual(f["cur_months_zero"][0], 2)
        self.assertAlmostEqual(f["cur_last_month_rate"][0], 0.0)
        self.assertAlmostEqual(f["cur_trend"][0], 0.0 - 20 / 22)

    def test_never_attended(self):
        f = feats([[0, 0]], [[22, 22]], ["jun", "jul"])
        self.assertTrue(np.isnan(f["cur_last_att_month"][0]))
        self.assertEqual(f["cur_trailing_zero_months"][0], 2)
        self.assertEqual(f["cur_any_attendance"][0], 0)

    def test_zero_working_days_is_no_data(self):
        # clean.py turns 0 working days into NaN; a NaN month is not reported and not a zero month
        f = feats([[20, NAN]], [[22, NAN]], ["jun", "jul"])
        self.assertEqual(f["cur_months_reported"][0], 1)
        self.assertEqual(f["cur_months_zero"][0], 0)
        self.assertTrue(np.isnan(f["cur_last_month_rate"][0]))
        self.assertEqual(f["cur_trailing_zero_months"][0], 1)

    def test_quarter_rates_use_visible_months_only(self):
        months = ["jun", "jul", "aug", "sep"]
        f = feats([[11, 11, 22, 0]], [[22, 22, 22, 22]], months)
        self.assertAlmostEqual(f["cur_q1_rate"][0], 44 / 66)
        self.assertAlmostEqual(f["cur_q2_rate"][0], 0.0)
        self.assertTrue(np.isnan(f["cur_q3_rate"][0]))
        self.assertTrue(np.isnan(f["cur_q4_rate"][0]))

    def test_no_data_at_all(self):
        f = feats([[NAN, NAN]], [[NAN, NAN]], ["jun", "jul"])
        self.assertTrue(np.isnan(f["cur_att_rate"][0]))
        self.assertEqual(f["cur_months_reported"][0], 0)
        self.assertTrue(np.isnan(f["cur_trend"][0]))

    def test_empty_month_list(self):
        f = attendance.attendance_features(np.zeros((3, 0)), np.zeros((3, 0)), [], "cur_")
        self.assertEqual(len(f["cur_att_rate"]), 3)
        self.assertTrue(np.isnan(f["cur_att_rate"]).all())
        self.assertTrue((f["cur_months_reported"] == 0).all())

    def test_month_arrays_reads_clean_columns(self):
        df = pd.DataFrame({"att_jun": [1.0], "wd_jun": [2.0], "att_jul": [3.0], "wd_jul": [4.0]})
        att, wd = attendance.month_arrays(df, ["jun", "jul"])
        self.assertEqual(att.tolist(), [[1.0, 3.0]])
        self.assertEqual(wd.tolist(), [[2.0, 4.0]])


if __name__ == "__main__":
    unittest.main()
