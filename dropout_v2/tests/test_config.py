from __future__ import annotations

import unittest

from dropout_v2 import config


class TestConfig(unittest.TestCase):
    def test_visible_months_follow_academic_calendar(self):
        self.assertEqual(config.visible_months("start"), [])
        self.assertEqual(config.visible_months("jul"), ["jun", "jul"])
        self.assertEqual(config.visible_months("apr")[-1], "apr")
        self.assertEqual(len(config.visible_months("apr")), 11)

    def test_unknown_snapshot_raises(self):
        with self.assertRaises(ValueError):
            config.visible_months("dec")

    def test_year_navigation(self):
        self.assertEqual(config.prev_year("2024-25"), "2023-24")
        self.assertEqual(config.prev_year("2024-25", 2), "2022-23")
        self.assertIsNone(config.prev_year("2022-23"))
        self.assertEqual(config.next_year("2025-26"), "2026-27")
        self.assertIsNone(config.next_year("2026-27"))
        self.assertEqual(config.year_start("2025-26"), 2025)

    def test_base_classes_exclude_class_10(self):
        self.assertEqual(config.BASE_CLASSES, (1, 2, 3, 4, 5, 6, 7, 8, 9))

    def test_split_years(self):
        self.assertEqual(config.TRAIN_YEARS, ["2023-24"])
        self.assertEqual(config.VALID_YEAR, "2024-25")
        self.assertEqual(config.TEST_YEAR, "2025-26")
        self.assertEqual(config.LIVE_YEAR, "2026-27")


if __name__ == "__main__":
    unittest.main()
