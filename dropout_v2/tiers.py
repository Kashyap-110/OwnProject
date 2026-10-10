"""Risk tiers by statewide rank among students who are attending (or whose status is not yet known)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from dropout_v2 import config

TIERED_STATUSES = ("ATTENDING", "UNKNOWN")


def assign_tiers(df: pd.DataFrame, prob_col: str, status_col: str = "engagement_status",
                 district_col: str = "district") -> pd.DataFrame:
    """Add risk_tier, state_pct and district_pct.

    state_pct = rank / n among tiered students (0.001 = top 0.1%). Tier = first cut-off with
    (rank - 1) / n < cut. NOT_ATTENDING and NO_SCHOOL_DATA students keep their status as tier.
    """
    out = df.copy()
    status = out[status_col].astype(str)
    eligible = status.isin(TIERED_STATUSES).to_numpy()
    prob = out[prob_col].to_numpy(dtype="float64")

    state_rank = np.full(len(out), np.nan)
    n = int(eligible.sum())
    if n:
        order = np.argsort(-prob[eligible], kind="mergesort")
        ranks = np.empty(n)
        ranks[order] = np.arange(1, n + 1)
        state_rank[eligible] = ranks
    out["state_pct"] = state_rank / n if n else np.nan

    tier = np.where(eligible, config.TIER_REST, status.to_numpy())
    position = (state_rank - 1) / n if n else state_rank
    assigned = np.zeros(len(out), dtype=bool)
    for name, cut in config.TIER_CUTS:
        hit = eligible & ~assigned & (position < cut)
        tier = np.where(hit, name, tier)
        assigned |= hit
    out["risk_tier"] = tier

    district_pct = np.full(len(out), np.nan)
    if n:
        sub = out.loc[eligible, [district_col]].copy()
        sub["prob"] = prob[eligible]
        ranks = sub.groupby(district_col, observed=True)["prob"].rank(method="first", ascending=False)
        sizes = sub.groupby(district_col, observed=True)["prob"].transform("size")
        district_pct[eligible] = (ranks / sizes).to_numpy()
    out["district_pct"] = district_pct
    return out
