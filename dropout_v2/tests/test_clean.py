from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from dropout_v2 import clean, config
from dropout_v2.tests.helpers import no_attendance, raw_frame, raw_row, set_months


class TestToKey(unittest.TestCase):
    def test_digits_kept_junk_minus_one(self):
        values = pd.Series(["096904697373", " 123 ", "abc", None, "12a", "999999999999"])
        self.assertEqual(clean.to_key(values).tolist(), [96904697373, 123, -1, -1, -1, 999999999999])


class TestCleanText(unittest.TestCase):
    def test_null_tokens_case_and_spaces(self):
        values = pd.Series([" male ", "null", "", None, "Female", "N/A"])
        out = clean.clean_text(values)
        self.assertEqual(list(out.astype(str)), ["MALE", "UNKNOWN", "UNKNOWN", "UNKNOWN", "FEMALE", "UNKNOWN"])
        self.assertIsInstance(out.dtype, pd.CategoricalDtype)

    def test_mapping_with_default(self):
        values = pd.Series(["BPL", "aay", "APL", "null"])
        out = clean.clean_text(values, {"BPL": "BPL", "AAY": "AAY"}, default="OTHER")
        self.assertEqual(list(out.astype(str)), ["BPL", "AAY", "OTHER", "UNKNOWN"])

    def test_categorical_input(self):
        values = pd.Series(["R", "U", "r", None], dtype="category")
        out = clean.clean_text(values, clean.TEXT_MAPS["rural_urban_flag"])
        self.assertEqual(list(out.astype(str)), ["RURAL", "URBAN", "RURAL", "UNKNOWN"])


class TestCleanYear(unittest.TestCase):
    def clean(self, rows, year="2024-25"):
        return clean.clean_year(raw_frame(rows), year)

    def test_placeholder_rows_removed_and_recorded(self):
        rows = [raw_row(person_id="999999999999", student_id=str(i)) for i in range(5)]
        rows.append(raw_row(person_id="000000000002"))
        out, diag = self.clean(rows)
        self.assertEqual(out["person_key"].tolist(), [2])
        self.assertEqual(diag["placeholder_ids"], {"999999999999": 5})
        self.assertEqual(diag["rows_placeholder"], 5)

    def test_bad_person_id_dropped(self):
        out, diag = self.clean([raw_row(person_id="abc"), raw_row(person_id="5")])
        self.assertEqual(out["person_key"].tolist(), [5])
        self.assertEqual(diag["rows_bad_person_id"], 1)

    def test_duplicates_collapse_to_one_row(self):
        # School A: Jun-Oct (5 months x 20 = 100 days). School B: Nov-Apr (6 months x 15 = 90 days).
        first = {m: (20.0 if i < 5 else None) for i, m in enumerate(config.MONTHS)}
        first_wd = {m: (22.0 if i < 5 else None) for i, m in enumerate(config.MONTHS)}
        second = {m: (15.0 if 5 <= i <= 10 else None) for i, m in enumerate(config.MONTHS)}
        second_wd = {m: (22.0 if 5 <= i <= 10 else None) for i, m in enumerate(config.MONTHS)}
        a = set_months(raw_row(school_id="28000000001"), first, first_wd)
        b = set_months(raw_row(school_id="28000000002"), second, second_wd)
        out, diag = self.clean([b, a])
        self.assertEqual(len(out), 1)
        row = out.iloc[0]
        self.assertEqual(int(row["n_rows_in_year"]), 2)
        self.assertEqual(int(row["n_schools_in_year"]), 2)
        self.assertEqual(float(row["att_jun"]), 20.0)
        self.assertEqual(float(row["att_apr"]), 15.0)
        # the main row is the one with more attended days: school A
        self.assertEqual(int(row["school_key"]), 28000000001)
        self.assertEqual(diag["persons_with_duplicates"], 1)

    def test_class_range(self):
        rows = [raw_row(person_id=str(i + 1), class_currently_enrolled=c) for i, c in enumerate([-4, -3, 0, 12, 13, 5])]
        out, diag = self.clean(rows)
        self.assertEqual(sorted(out["class_t"].tolist()), [-3, 5, 12])
        self.assertEqual(diag["rows_bad_class"], 3)

    def test_attendance_rules(self):
        row = set_months(raw_row(), {"jun": 30.0, "jul": 10.0, "aug": 5.0}, {"jun": 22.0, "jul": 0.0, "aug": 40.0})
        out, _ = self.clean([row])
        r = out.iloc[0]
        self.assertEqual(float(r["att_jun"]), 22.0)           # clipped to working days
        self.assertTrue(np.isnan(r["wd_jul"]) and np.isnan(r["att_jul"]))   # 0 working days = no data
        self.assertTrue(np.isnan(r["wd_aug"]) and np.isnan(r["att_aug"]))   # > 31 working days = invalid

    def test_text_cleaning_and_flags(self):
        rows = [
            raw_row(person_id="1", student_gender="Transgender", social_category="null", rural_urban_flag="U",
                    is_deceased="Deceased"),
            raw_row(person_id="2", student_gender="Others", is_deceased="reported deceased"),
        ]
        out, _ = self.clean(rows)
        out = out.set_index("person_key")
        self.assertEqual(out.loc[1, "gender"], "OTHER")
        self.assertEqual(out.loc[2, "gender"], "OTHER")
        self.assertEqual(out.loc[1, "social_category"], "UNKNOWN")
        self.assertEqual(out.loc[1, "rural_urban"], "URBAN")
        self.assertTrue(bool(out.loc[1, "deceased"]))
        self.assertFalse(bool(out.loc[2, "deceased"]))

    def test_person_id_string_kept(self):
        out, _ = self.clean([raw_row(person_id="096904697373")])
        self.assertEqual(out["person_id"].iloc[0], "096904697373")
        self.assertEqual(int(out["person_key"].iloc[0]), 96904697373)

    def test_marks_outlier_rank_and_zero(self):
        rows = [raw_row(person_id=str(i + 1), fa_marks=400.0 + i, sa_marks=200.0 + i) for i in range(200)]
        rows.append(raw_row(person_id="901", fa_marks=50000.0))
        rows.append(raw_row(person_id="902", fa_marks=0.0, sa_marks=None))
        out, _ = self.clean(rows)
        out = out.set_index("person_key")
        self.assertTrue(np.isnan(out.loc[901, "fa_marks"]))
        self.assertEqual(int(out.loc[902, "fa_zero"]), 1)
        self.assertTrue(np.isnan(out.loc[902, "fa_rank"]))
        ranks = out["fa_rank"].dropna()
        self.assertTrue(((ranks > 0) & (ranks <= 1)).all())
        self.assertGreater(out.loc[200, "fa_rank"], out.loc[1, "fa_rank"])

    def test_coordinates_out_of_range(self):
        out, _ = self.clean([raw_row(house_latitude=40.0, house_longitude=81.0)])
        self.assertTrue(np.isnan(out["house_lat"].iloc[0]))

    def test_clean_output_dtypes_are_compact(self):
        out, _ = self.clean([raw_row(person_id=str(i + 1)) for i in range(3)])
        self.assertEqual(out["att_jun"].dtype, np.float32)
        self.assertEqual(out["wd_jun"].dtype, np.float32)
        self.assertEqual(out["class_t"].dtype, np.int8)
        self.assertEqual(out["person_key"].dtype, np.int64)
        for col in ["gender", "social_category", "district", "mandal", "medium"]:
            self.assertIsInstance(out[col].dtype, pd.CategoricalDtype, col)
        self.assertEqual(list(out.columns), clean.CLEAN_COLUMNS)

    def test_dob_parsed(self):
        out, _ = self.clean([raw_row(student_date_of_birth="2016-03-01"),
                             raw_row(person_id="2", student_date_of_birth="not a date")])
        out = out.set_index("person_key")
        self.assertEqual(out.loc[1, "dob"], pd.Timestamp("2016-03-01"))
        self.assertTrue(pd.isna(out.loc[2, "dob"]))


if __name__ == "__main__":
    unittest.main()
