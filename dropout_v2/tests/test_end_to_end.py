from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from dropout_v2 import config, synthetic, train


class TestTrainingEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.data_root = cls.tmp / "data"
        cls.art_root = cls.tmp / "artifacts"
        synthetic.write_synthetic_cache(tag="synthetic", n_students=6000, seed=11, root=cls.data_root)
        cls.run_dir = train.run_training(
            tag="synthetic", snapshots=["start", "jul", "apr"], bootstrap_reps=5,
            data_root=cls.data_root, artifact_root=cls.art_root)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(str(cls.tmp), ignore_errors=True)

    def test_artifacts_written(self):
        for name in ["config.json", "run.log", "data_profile.md", "label_report.csv", "label_summary.csv",
                     "MODEL_REPORT.md"]:
            self.assertTrue((self.run_dir / name).exists(), name)
        for snap in ["start", "jul", "apr"]:
            for name in ["bundle_{}.joblib", "bundle_{}_eval.joblib", "model_{}.txt", "metrics_{}.json",
                         "topk_{}.csv", "tiers_{}.csv", "importance_{}.csv", "pr_curve_{}.png",
                         "calibration_{}.png", "breakdown_{}_class_t.csv"]:
                self.assertTrue((self.run_dir / name.format(snap)).exists(), name.format(snap))

    def test_jul_model_beats_chance_on_test_year(self):
        metrics = json.loads((self.run_dir / "metrics_jul.json").read_text(encoding="utf-8"))
        self.assertEqual(metrics["test_year"], config.TEST_YEAR)
        self.assertGreater(metrics["test"]["auc"], 0.7)
        self.assertIn("attending", metrics)
        self.assertIn("rule_baseline", metrics)
        self.assertIn("bootstrap_ci", metrics)
        self.assertLessEqual(metrics["bootstrap_ci"]["auc"][0], metrics["bootstrap_ci"]["auc"][1])

    def test_report_mentions_every_snapshot(self):
        text = (self.run_dir / "MODEL_REPORT.md").read_text(encoding="utf-8")
        for snap in ["start", "jul", "apr"]:
            self.assertIn("Snapshot `{}`".format(snap), text)
        self.assertIn("Known limitations", text)

    def test_latest_pointer_updated(self):
        latest = (self.art_root / "LATEST").read_text(encoding="utf-8").strip()
        self.assertEqual(Path(latest), self.run_dir.resolve())

    def test_production_bundle_calibrated_on_test_year(self):
        from dropout_v2 import modeling
        prod = modeling.load_bundle(self.run_dir / "bundle_jul.joblib")
        self.assertEqual(prod.calibration_year, config.TEST_YEAR)
        self.assertIn(config.VALID_YEAR, prod.train_years)


if __name__ == "__main__":
    unittest.main()
