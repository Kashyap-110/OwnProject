from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from dropout_v2 import config, schema, synthetic


class TestSynthetic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.years, cls.schools = synthetic.make_raw_years(n_students=1500, seed=1)

    def test_all_years_present_with_raw_columns(self):
        self.assertEqual(sorted(self.years), sorted(config.ACADEMIC_YEARS))
        for frame in self.years.values():
            self.assertEqual(list(frame.columns), schema.RAW_COLUMNS)
        self.assertEqual(list(self.schools.columns), schema.SCHOOL_COLUMNS)

    def test_placeholder_person_id_in_2024_25(self):
        counts = self.years["2024-25"]["person_id"].value_counts()
        self.assertEqual(int(counts.get(synthetic.PLACEHOLDER_ID, 0)), 40)

    def test_live_year_has_only_june_july_attendance(self):
        live = self.years[config.LIVE_YEAR]
        for month in config.MONTHS[2:]:
            self.assertTrue((live[month + "_working_days"].fillna(0) == 0).all(), month)
        self.assertTrue(live["fa_marks"].isna().all())

    def test_person_ids_have_leading_zeros(self):
        ids = self.years["2025-26"]["person_id"]
        self.assertTrue(ids.str.startswith("0").any())

    def test_2023_24_class_1_mostly_missing(self):
        c22 = (self.years["2022-23"]["class_currently_enrolled"] == 1).sum()
        c23 = (self.years["2023-24"]["class_currently_enrolled"] == 1).sum()
        self.assertLess(c23, 0.3 * c22)

    def test_write_cache_matches_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = synthetic.write_synthetic_cache(tag="synthetic", n_students=300, seed=2, root=Path(tmp))
            table = pq.read_table(str(out / "raw" / "2024-25.parquet"))
            self.assertEqual(table.schema.names, schema.RAW_COLUMNS)
            self.assertTrue((out / "raw" / "schools.parquet").exists())


if __name__ == "__main__":
    unittest.main()
