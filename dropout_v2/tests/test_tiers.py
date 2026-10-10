from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from dropout_v2 import tiers


class TestTiers(unittest.TestCase):
    def frame(self):
        n = 1000
        df = pd.DataFrame({
            "risk_probability": np.linspace(0.9, 0.001, n),
            "engagement_status": ["ATTENDING"] * n,
            "district": ["D1"] * 500 + ["D2"] * 500,
        })
        extra = pd.DataFrame({
            "risk_probability": [0.99, 0.98],
            "engagement_status": ["NOT_ATTENDING", "NO_SCHOOL_DATA"],
            "district": ["D1", "D2"],
        })
        return pd.concat([df, extra], ignore_index=True)

    def test_tier_shares_follow_cutoffs(self):
        out = tiers.assign_tiers(self.frame(), "risk_probability")
        counts = out["risk_tier"].value_counts().to_dict()
        self.assertEqual(counts["CRITICAL"], 20)
        self.assertEqual(counts["HIGH"], 50)
        self.assertEqual(counts["MEDIUM"], 100)
        self.assertEqual(counts["LOW"], 830)

    def test_non_attending_keep_own_tier(self):
        out = tiers.assign_tiers(self.frame(), "risk_probability")
        self.assertEqual(out.loc[1000, "risk_tier"], "NOT_ATTENDING")
        self.assertEqual(out.loc[1001, "risk_tier"], "NO_SCHOOL_DATA")
        self.assertTrue(np.isnan(out.loc[1000, "state_pct"]))

    def test_percentiles(self):
        out = tiers.assign_tiers(self.frame(), "risk_probability")
        self.assertAlmostEqual(out.loc[0, "state_pct"], 0.001)
        self.assertAlmostEqual(out.loc[500, "district_pct"], 1 / 500)   # top of district D2
        self.assertEqual(out.loc[0, "risk_tier"], "CRITICAL")

    def test_unknown_status_is_tiered(self):
        df = pd.DataFrame({"risk_probability": [0.5, 0.1], "engagement_status": ["UNKNOWN", "UNKNOWN"],
                           "district": ["D1", "D1"]})
        out = tiers.assign_tiers(df, "risk_probability")
        self.assertEqual(out["risk_tier"].tolist(), ["CRITICAL", "LOW"])


if __name__ == "__main__":
    unittest.main()
