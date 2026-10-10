from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from dropout_v2 import config, score, synthetic, train


class TestMonthDetection(unittest.TestCase):
    def frame(self, coverage):
        n = 100
        data = {}
        for i, month in enumerate(config.MONTHS):
            share = coverage[i] if i < len(coverage) else 0.0
            has = np.arange(n) < int(share * n)
            data["wd_" + month] = np.where(has, 22.0, np.nan)
            data["att_" + month] = np.where(has, 20.0, np.nan)
        return pd.DataFrame(data)

    def test_detect_available_months_ignores_zero_months(self):
        # clean.py turns the live year's 0/0 future months into NaN, so they have no coverage
        self.assertEqual(score.detect_available_months(self.frame([0.85, 0.84])), 2)

    def test_detection_stops_at_first_thin_month(self):
        self.assertEqual(score.detect_available_months(self.frame([0.9, 0.9, 0.2, 0.9])), 2)
        self.assertEqual(score.detect_available_months(self.frame([0.1])), 0)

    def test_choose_snapshot(self):
        trained = list(config.SNAPSHOTS)
        self.assertEqual(score.choose_snapshot(2, trained), "jul")
        self.assertEqual(score.choose_snapshot(1, trained), "start")
        self.assertEqual(score.choose_snapshot(7, trained), "nov")
        self.assertEqual(score.choose_snapshot(11, trained), "apr")
        with self.assertRaises(ValueError):
            score.choose_snapshot(1, ["jul"])


class TestScoringEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.data_root = cls.tmp / "data"
        cls.art_root = cls.tmp / "artifacts"
        synthetic.write_synthetic_cache(tag="synthetic", n_students=5000, seed=21, root=cls.data_root)
        cls.run_dir = train.run_training(tag="synthetic", snapshots=["start", "jul"], bootstrap_reps=0,
                                         baselines=False, data_root=cls.data_root, artifact_root=cls.art_root)
        cls.legacy_path = cls.tmp / "legacy" / "statewide_scores.parquet"
        cls.legacy_path.parent.mkdir(parents=True)
        pd.DataFrame({"old": [1]}).to_parquet(str(cls.legacy_path))
        cls.out_dir = score.run_scoring(tag="synthetic", snapshot="auto", legacy_export=True,
                                        data_root=cls.data_root, artifact_root=cls.art_root,
                                        legacy_path=cls.legacy_path)
        cls.scores = pd.read_parquet(str(cls.out_dir / "scores_{}_jul.parquet".format(config.LIVE_YEAR)))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(str(cls.tmp), ignore_errors=True)

    def test_auto_snapshot_is_jul(self):
        self.assertTrue((self.scores["snapshot"] == "jul").all())

    def test_scores_columns_and_ranges(self):
        for col in ["person_id", "student_id", "school_id", "school_name", "district", "mandal", "class_t",
                    "engagement_status", "risk_probability", "risk_tier", "state_pct", "district_pct",
                    "reason_1", "reason_2", "reason_3", "cur_att_rate", "prev_att_rate"]:
            self.assertIn(col, self.scores.columns)
        self.assertTrue(self.scores["risk_probability"].between(0, 1).all())
        self.assertTrue(self.scores["class_t"].between(1, 9).all())
        self.assertTrue(self.scores["person_id"].astype(str).str.len().ge(1).all())

    def test_tiers_present(self):
        tiers_seen = set(self.scores["risk_tier"])
        self.assertTrue({"CRITICAL", "HIGH", "MEDIUM", "LOW"} <= tiers_seen)

    def test_output_files(self):
        for name in ["district_summary.csv", "top_risk_by_district.csv", "not_attending.csv", "SCORING_SUMMARY.md",
                     "scoring_info.json"]:
            self.assertTrue((self.out_dir / name).exists(), name)
        district = pd.read_csv(self.out_dir / "district_summary.csv")
        self.assertIn("expected_dropouts", district.columns)
        top = pd.read_csv(self.out_dir / "top_risk_by_district.csv")
        self.assertFalse(top["risk_tier"].isin(["NOT_ATTENDING", "NO_SCHOOL_DATA"]).any())

    def test_legacy_export_for_dashboard(self):
        legacy = pd.read_parquet(str(self.legacy_path))
        for col in ["person_id", "school_name", "baseline_class", "student_gender", "student_district_name",
                    "social_category", "dwelling_type", "ml_risk_score", "ml_risk_level", "risk_tier_v2"]:
            self.assertIn(col, legacy.columns)
        self.assertNotIn("stakeholder_manual_score", legacy.columns)
        self.assertTrue(set(legacy["ml_risk_level"]) <= {"High", "Medium", "Low"})
        backups = list(self.legacy_path.parent.glob("statewide_scores.backup_*.parquet"))
        self.assertEqual(len(backups), 1)


if __name__ == "__main__":
    unittest.main()
