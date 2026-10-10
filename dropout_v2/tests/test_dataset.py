from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from dropout_v2 import config, features, synthetic
from dropout_v2.dataset import DataStore


class TestDataStore(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        synthetic.write_synthetic_cache(tag="synthetic", n_students=800, seed=7, root=cls.tmp)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(str(cls.tmp), ignore_errors=True)

    def store(self):
        return DataStore("synthetic", root=self.tmp)

    def test_clean_cached_and_reloaded_with_same_dtypes(self):
        first = self.store().clean("2024-25")
        self.assertTrue(any(self.store().derived_dir.glob("clean/2024-25.parquet")))
        second = self.store().clean("2024-25")
        self.assertEqual(list(first.dtypes.astype(str)), list(second.dtypes.astype(str)))
        self.assertEqual(len(first), len(second))

    def test_years_available(self):
        self.assertEqual(self.store().years_available(), config.ACADEMIC_YEARS)

    def test_labels_absent_for_live_year(self):
        self.assertIsNone(self.store().labels(config.LIVE_YEAR))

    def test_labels_have_reporting_columns(self):
        lab = self.store().labels("2024-25")
        for col in ["person_key", "label", "label_reason", "school_key", "mandal_key", "class_t", "src_is_dropout"]:
            self.assertIn(col, lab.columns)
        base = features.base_population(self.store().clean("2024-25"))
        self.assertEqual(len(lab), len(base))

    def test_dataset_has_label_and_features(self):
        ds = self.store().dataset("2024-25", "jul")
        numeric, categorical = features.feature_columns("jul")
        for col in numeric + categorical + ["label", "label_reason"]:
            self.assertIn(col, ds.columns)
        self.assertEqual(len(ds), len(self.store().labels("2024-25")))
        live = self.store().dataset(config.LIVE_YEAR, "jul", with_label=False)
        self.assertNotIn("label", live.columns)

    def test_missing_raw_file_raises_with_hint(self):
        empty = Path(tempfile.mkdtemp())
        try:
            with self.assertRaises(FileNotFoundError) as ctx:
                DataStore("full", root=empty).clean("2024-25")
            self.assertIn("extract", str(ctx.exception))
        finally:
            shutil.rmtree(str(empty), ignore_errors=True)

    def test_signature_changes_with_label_rules(self):
        before = self.store().derived_dir
        with mock.patch.object(config, "LABEL_WINDOW", ["jun", "jul", "aug"]):
            after = self.store().derived_dir
        self.assertNotEqual(before, after)

    def test_refreshed_raw_file_rebuilds_derived_data(self):
        import os
        import pyarrow.parquet as pq

        store = self.store()
        before_clean = len(store.clean("2024-25"))
        before_ds = len(store.dataset("2024-25", "jul"))
        raw_path = store.raw_path("2024-25")
        table = pq.read_table(str(raw_path))
        refreshed = table.slice(0, table.num_rows // 2)
        backup = raw_path.with_name("backup.parquet")
        pq.write_table(table, str(backup))
        try:
            pq.write_table(refreshed, str(raw_path))
            future = os.path.getmtime(str(raw_path)) + 60
            os.utime(str(raw_path), (future, future))
            fresh = self.store()
            self.assertLess(len(fresh.clean("2024-25")), before_clean)
            self.assertLess(len(fresh.dataset("2024-25", "jul")), before_ds)
        finally:
            pq.write_table(table, str(raw_path))
            later = os.path.getmtime(str(raw_path)) + 120
            os.utime(str(raw_path), (later, later))
            backup.unlink()

    def test_dataset_numeric_features_are_float32(self):
        ds = self.store().dataset("2024-25", "jul")
        numeric, _ = features.feature_columns("jul")
        wide = [c for c in numeric if ds[c].dtype != "float32"]
        self.assertEqual(wide, [])

    def test_profile_table(self):
        store = self.store()
        for year in config.ACADEMIC_YEARS:
            store.clean(year)
        profile = store.profile()
        self.assertEqual(profile["academic_year"].tolist(), config.ACADEMIC_YEARS)
        self.assertIn("rows_placeholder", profile.columns)
        self.assertEqual(int(profile.set_index("academic_year").loc["2024-25", "rows_placeholder"]), 40)


if __name__ == "__main__":
    unittest.main()
