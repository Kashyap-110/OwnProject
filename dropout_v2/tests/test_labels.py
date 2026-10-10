from __future__ import annotations

import unittest

import numpy as np

from dropout_v2 import clean, config, labels
from dropout_v2.tests.helpers import no_attendance, raw_frame, raw_row, set_months

T, T1 = "2024-25", "2025-26"
REPORTING_SCHOOL = "28000000001"
SILENT_SCHOOL = "28000000009"


def cleaned(rows, year):
    frame, _ = clean.clean_year(raw_frame(rows), year)
    return frame


def classmates(n=4, school=REPORTING_SCHOOL, start=500):
    """Attending students that make a school 'reporting' in T+1."""
    return [raw_row(person_id=str(start + i), student_id=str(9000 + start + i), school_id=school,
                    academic_year=T1) for i in range(n)]


class TestLabels(unittest.TestCase):
    def run_case(self, base_rows, next_rows):
        base = cleaned(base_rows, T)
        nxt = cleaned(next_rows, T1)
        out = labels.build_labels(base, nxt)
        return out.set_index("person_key")

    def test_continued_attending(self):
        out = self.run_case([raw_row(person_id="1")], [raw_row(person_id="1", academic_year=T1)] + classmates())
        self.assertEqual(out.loc[1, "label"], 0)
        self.assertEqual(out.loc[1, "label_reason"], "CONTINUED_ATTENDING")

    def test_continued_class_11_plus(self):
        nxt = [no_attendance(raw_row(person_id="1", class_currently_enrolled=11, school_id=SILENT_SCHOOL))]
        out = self.run_case([raw_row(person_id="1", class_currently_enrolled=9)], nxt)
        self.assertEqual(out.loc[1, "label"], 0)
        self.assertEqual(out.loc[1, "label_reason"], "CONTINUED_CLASS_11_PLUS")

    def test_dropout_not_found(self):
        out = self.run_case([raw_row(person_id="1")], classmates())
        self.assertEqual(out.loc[1, "label"], 1)
        self.assertEqual(out.loc[1, "label_reason"], "DROPOUT_NOT_FOUND")

    def test_dropout_not_attending_in_reporting_school(self):
        nxt = [no_attendance(raw_row(person_id="1", school_id=REPORTING_SCHOOL))] + classmates()
        out = self.run_case([raw_row(person_id="1")], nxt)
        self.assertEqual(out.loc[1, "label"], 1)
        self.assertEqual(out.loc[1, "label_reason"], "DROPOUT_NOT_ATTENDING")

    def test_unknown_when_school_not_reporting(self):
        nxt = [no_attendance(raw_row(person_id="1", school_id=SILENT_SCHOOL)),
               no_attendance(raw_row(person_id="2", school_id=SILENT_SCHOOL))]
        out = self.run_case([raw_row(person_id="1")], nxt)
        self.assertTrue(np.isnan(out.loc[1, "label"]))
        self.assertEqual(out.loc[1, "label_reason"], "UNKNOWN_SCHOOL_NOT_REPORTING")

    def test_student_id_rescue(self):
        base = [raw_row(person_id="1", student_id="777")]
        nxt = [raw_row(person_id="2", student_id="777")] + classmates()
        out = self.run_case(base, nxt)
        self.assertEqual(out.loc[1, "label"], 0)
        self.assertEqual(out.loc[1, "label_reason"], "CONTINUED_ATTENDING")

    def test_august_only_is_not_evidence_for_jun_jul_window(self):
        aug_only = {m: (20.0 if m == "aug" else None) for m in config.MONTHS}
        aug_wd = {m: (22.0 if m == "aug" else None) for m in config.MONTHS}
        nxt = [set_months(raw_row(person_id="1"), aug_only, aug_wd)] + classmates()
        out = self.run_case([raw_row(person_id="1")], nxt)
        self.assertEqual(out.loc[1, "label"], 1)
        self.assertEqual(out.loc[1, "label_reason"], "DROPOUT_NOT_ATTENDING")

    def test_evidence_wins_over_class_11(self):
        nxt = [raw_row(person_id="1", class_currently_enrolled=11)] + classmates()
        out = self.run_case([raw_row(person_id="1", class_currently_enrolled=9)], nxt)
        self.assertEqual(out.loc[1, "label_reason"], "CONTINUED_ATTENDING")

    def test_output_aligned_with_base(self):
        base = cleaned([raw_row(person_id=str(i + 1), student_id=str(100 + i)) for i in range(3)], T)
        nxt = cleaned([raw_row(person_id="2", student_id="101")] + classmates(), T1)
        out = labels.build_labels(base, nxt)
        self.assertEqual(out["person_key"].tolist(), base["person_key"].tolist())
        self.assertEqual(out["label"].tolist(), [1.0, 0.0, 1.0])


class TestSchoolReporting(unittest.TestCase):
    def test_share_per_school(self):
        rows = classmates(n=3) + [no_attendance(raw_row(person_id="99", school_id=REPORTING_SCHOOL))]
        rows += [no_attendance(raw_row(person_id="98", school_id=SILENT_SCHOOL))]
        frame = cleaned(rows, T1)
        share = labels.school_reporting_share(frame, ["jun", "jul"])
        self.assertAlmostEqual(share.loc[int(REPORTING_SCHOOL)], 0.75)
        self.assertAlmostEqual(share.loc[int(SILENT_SCHOOL)], 0.0)


if __name__ == "__main__":
    unittest.main()
