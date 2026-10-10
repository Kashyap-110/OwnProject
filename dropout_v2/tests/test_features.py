from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from dropout_v2 import clean, config, features, labels, synthetic
from dropout_v2.tests.helpers import no_attendance, raw_frame, raw_row, school_frame, set_months


def cleaned(rows, year):
    frame, _ = clean.clean_year(raw_frame(rows), year)
    return frame


def synthetic_frames(n=1200, seed=5):
    years, schools = synthetic.make_raw_years(n_students=n, seed=seed)
    frames = {y: clean.clean_year(f, y)[0] for y, f in years.items()}
    return frames, features.prepare_school_dim(schools)


def prev_labels_for(frames, year):
    prev = config.prev_year(year)
    base = frames[prev][frames[prev]["class_t"].isin(config.BASE_CLASSES)]
    lab = labels.build_labels(base, frames[year])
    lab["school_key"] = base["school_key"].to_numpy()
    lab["mandal_key"] = (base["district"].astype(str) + "|" + base["mandal"].astype(str)).to_numpy()
    return lab


class TestFeaturesSynthetic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frames, cls.dim = synthetic_frames()

    def build(self, year, snapshot, frames=None):
        frames = frames or self.frames
        prev = prev_labels_for(frames, year) if config.prev_year(year) in frames else None
        return features.build_features(year, snapshot, frames, self.dim, prev)

    def test_only_base_classes_and_no_deceased(self):
        out = self.build("2024-25", "jul")
        self.assertTrue(out["class_t"].isin(config.BASE_CLASSES).all())
        dead = self.frames["2024-25"].loc[self.frames["2024-25"]["deceased"], "person_key"]
        self.assertFalse(out["person_key"].isin(dead).any())

    def test_snapshot_features_ignore_future(self):
        for snapshot in ("jul", "nov"):
            before = self.build("2024-25", snapshot)
            changed = {y: f.copy() for y, f in self.frames.items()}
            t = changed["2024-25"]
            hidden = config.MONTHS[config.SNAPSHOTS[snapshot]:]
            for m in hidden:
                t["att_" + m] = np.float32(0.0)
                t["wd_" + m] = np.float32(5.0)
            t["fa_marks"] = np.float32(1.0)
            t["fa_rank"] = np.float32(0.01)
            t["sa_rank"] = np.float32(0.01)
            t["n_rows_in_year"] = np.int16(7)
            for future in ("2025-26", "2026-27"):
                f = changed[future]
                for m in config.MONTHS:
                    f["att_" + m] = np.float32(0.0)
                f["class_t"] = np.int8(1)
            # previous-year labels depend only on Jun-Jul of 2024-25, which every snapshot here can see
            after = features.build_features("2024-25", snapshot, changed, self.dim, prev_labels_for(self.frames, "2024-25"))
            pd.testing.assert_frame_equal(before.reset_index(drop=True), after.reset_index(drop=True))

    def test_feature_columns_exist_for_every_snapshot(self):
        for snapshot in config.SNAPSHOTS:
            out = self.build("2024-25", snapshot)
            numeric, categorical = features.feature_columns(snapshot)
            missing = [c for c in numeric + categorical if c not in out.columns]
            self.assertEqual(missing, [], snapshot)
            for c in categorical:
                self.assertIsInstance(out[c].dtype, pd.CategoricalDtype, c)

    def test_start_snapshot_has_no_current_year_or_rates(self):
        numeric, categorical = features.feature_columns("start")
        self.assertFalse(any(c.startswith("cur_") for c in numeric))
        self.assertNotIn("school_prev_dropout_rate", numeric)
        self.assertNotIn("engagement_status", categorical)
        out = self.build("2024-25", "start")
        self.assertTrue((out["engagement_status"].astype(str) == "UNKNOWN").all())

    def test_marks_of_current_year_only_at_apr(self):
        self.assertIn("cur_fa_rank", features.feature_columns("apr")[0])
        self.assertNotIn("cur_fa_rank", features.feature_columns("feb")[0])
        self.assertIn("prev_fa_rank", features.feature_columns("jul")[0])

    def test_rates_are_smoothed_toward_global(self):
        out = self.build("2024-25", "jul")
        rates = out["school_prev_dropout_rate"].dropna()
        self.assertTrue(((rates > 0) & (rates < 1)).all())

    def test_first_year_has_no_previous_year(self):
        out = self.build("2022-23", "jul")
        self.assertTrue((out["prev_status"].astype(str) == "PREV_YEAR_NOT_AVAILABLE").all())
        self.assertTrue(out["rejoiner_flag"].isna().all())
        self.assertTrue(out["school_prev_dropout_rate"].isna().all())


class TestFeaturesHandMade(unittest.TestCase):
    def setUp(self):
        self.dim = features.prepare_school_dim(school_frame([
            {"school_id": "28000000001", "highest_class": 5.0, "school_management": "MPP_ZPP SCHOOLS"},
            {"school_id": "28000000002", "highest_class": 10.0, "school_management": "PVT.UNAIDED (CBSE SYLLABUS)",
             "total_teachers": 0.0},
        ]))

    def test_prev_status_categories(self):
        prev = cleaned([
            raw_row(person_id="1", class_currently_enrolled=3, academic_year="2023-24"),
            raw_row(person_id="2", class_currently_enrolled=4, academic_year="2023-24"),
            raw_row(person_id="3", class_currently_enrolled=7, academic_year="2023-24"),
        ], "2023-24")
        cur = cleaned([
            raw_row(person_id="1", class_currently_enrolled=4),     # promoted
            raw_row(person_id="2", class_currently_enrolled=4),     # repeated
            raw_row(person_id="3", class_currently_enrolled=5),     # irregular
            raw_row(person_id="4", class_currently_enrolled=6),     # not in previous year
            raw_row(person_id="5", class_currently_enrolled=2),     # expected class 1 in 2023-24: known gap
            raw_row(person_id="6", class_currently_enrolled=1),     # expected pre-primary in 2023-24: known gap
        ], "2024-25")
        out = features.build_features("2024-25", "jul", {"2023-24": prev, "2024-25": cur}, self.dim, None)
        status = dict(zip(out["person_key"], out["prev_status"].astype(str)))
        self.assertEqual(status, {1: "PROMOTED", 2: "REPEATED", 3: "OTHER_CHANGE", 4: "NOT_IN_PREV_YEAR",
                                  5: "PREV_YEAR_DATA_GAP", 6: "PREV_YEAR_DATA_GAP"})
        same = dict(zip(out["person_key"], out["prev_same_school"]))
        self.assertEqual(same[1], 1.0)
        self.assertTrue(np.isnan(same[4]))

    def test_rejoiner_flag(self):
        y22 = cleaned([raw_row(person_id="1", class_currently_enrolled=3, academic_year="2022-23")], "2022-23")
        y23 = cleaned([raw_row(person_id="9", class_currently_enrolled=3, academic_year="2023-24")], "2023-24")
        y24 = cleaned([raw_row(person_id="1", class_currently_enrolled=5),
                       raw_row(person_id="2", class_currently_enrolled=5)], "2024-25")
        out = features.build_features("2024-25", "jul", {"2022-23": y22, "2023-24": y23, "2024-25": y24}, self.dim, None)
        flag = dict(zip(out["person_key"], out["rejoiner_flag"]))
        self.assertEqual(flag[1], 1.0)
        self.assertEqual(flag[2], 0.0)

    def test_engagement_status(self):
        cur = cleaned([
            raw_row(person_id="1"),                                                    # attends
            no_attendance(raw_row(person_id="2")),                                     # school reports, student none
            raw_row(person_id="3"), raw_row(person_id="4"),
            no_attendance(raw_row(person_id="5", school_id="28000000002")),            # school sends nothing
        ], "2024-25")
        out = features.build_features("2024-25", "jul", {"2024-25": cur}, self.dim, None)
        status = dict(zip(out["person_key"], out["engagement_status"].astype(str)))
        self.assertEqual(status[1], "ATTENDING")
        self.assertEqual(status[2], "NOT_ATTENDING")
        self.assertEqual(status[5], "NO_SCHOOL_DATA")

    def test_engagement_uses_latest_visible_month(self):
        june_only = set_months(raw_row(person_id="1"), {"jul": None}, {"jul": None})      # stopped after June
        july_zero = set_months(raw_row(person_id="2"), {"jul": 0.0})                      # on register, 0 days in July
        cur = cleaned([june_only, july_zero, raw_row(person_id="3"), raw_row(person_id="4"),
                       raw_row(person_id="5")], "2024-25")
        out = features.build_features("2024-25", "jul", {"2024-25": cur}, self.dim, None)
        status = dict(zip(out["person_key"], out["engagement_status"].astype(str)))
        self.assertEqual(status[1], "NOT_ATTENDING")
        self.assertEqual(status[2], "NOT_ATTENDING")
        self.assertEqual(status[3], "ATTENDING")

    def test_age_and_school_features(self):
        cur = cleaned([
            raw_row(person_id="1", class_currently_enrolled=5, student_date_of_birth="2014-06-15"),
            raw_row(person_id="2", class_currently_enrolled=5, student_date_of_birth="1963-01-01"),
            raw_row(person_id="3", class_currently_enrolled=4, school_id="28000000002"),
        ], "2024-25")
        out = features.build_features("2024-25", "jul", {"2024-25": cur}, self.dim, None).set_index("person_key")
        self.assertEqual(out.loc[1, "age_t"], 9.0)                  # on 2024-06-01 the student is 9
        self.assertEqual(out.loc[1, "over_age"], -1.0)
        self.assertTrue(np.isnan(out.loc[2, "age_t"]))              # 61 years old -> invalid
        self.assertEqual(out.loc[2, "dob_missing"], 1.0)
        self.assertEqual(out.loc[1, "is_terminal_class"], 1.0)      # class 5 in a 1-5 school
        self.assertEqual(out.loc[3, "is_terminal_class"], 0.0)
        self.assertEqual(out.loc[3, "mgmt_group"], "PRIVATE_UNAIDED_CBSE_ICSE")
        self.assertTrue(np.isnan(out.loc[3, "ptr"]))                # 0 teachers -> not known
        self.assertLess(out.loc[1, "home_school_km"], 5.0)


if __name__ == "__main__":
    unittest.main()
