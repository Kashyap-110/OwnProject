"""Monthly attendance -> features for a window of visible months.

Inputs are the cleaned monthly arrays: NaN means "no data for that month" (clean.py turns
0 or invalid working days into NaN), so a NaN month is neither attended nor a zero month.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from dropout_v2 import config


def month_arrays(df: pd.DataFrame, months: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """(attended, working days) arrays of shape (rows, len(months)), float64, NaN = no data."""
    n = len(df)
    if not months:
        return np.zeros((n, 0)), np.zeros((n, 0))
    att = df[["att_{}".format(m) for m in months]].to_numpy(dtype="float64")
    wd = df[["wd_{}".format(m) for m in months]].to_numpy(dtype="float64")
    return att, wd


def _pooled_rate(att: np.ndarray, wd: np.ndarray) -> np.ndarray:
    if att.shape[1] == 0:
        return np.full(att.shape[0], np.nan)
    days = np.nansum(att, axis=1)
    work = np.nansum(wd, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        rate = np.where(work > 0, days / np.where(work > 0, work, 1.0), np.nan)
    return rate


def attendance_features(att: np.ndarray, wd: np.ndarray, months: List[str], prefix: str) -> Dict[str, np.ndarray]:
    """Attendance features for the given months. Keys start with `prefix`."""
    n, k = att.shape
    reported = ~np.isnan(wd) if k else np.zeros((n, 0), dtype=bool)
    attended = np.nan_to_num(att, nan=0.0) > 0 if k else np.zeros((n, 0), dtype=bool)
    with np.errstate(invalid="ignore", divide="ignore"):
        monthly_rate = np.where(reported, att / np.where(reported, wd, 1.0), np.nan) if k else np.zeros((n, 0))

    days = np.nansum(att, axis=1) if k else np.zeros(n)
    work = np.nansum(wd, axis=1) if k else np.zeros(n)
    months_reported = reported.sum(axis=1)
    months_zero = (reported & ~attended).sum(axis=1)

    trailing = np.zeros(n, dtype="float64")
    still = np.ones(n, dtype=bool)
    for j in range(k - 1, -1, -1):
        still &= ~attended[:, j]
        trailing += still

    last_att = np.full(n, np.nan)
    first_rate = np.full(n, np.nan)
    for j in range(k):
        last_att = np.where(attended[:, j], j + 1, last_att)
        first_rate = np.where(np.isnan(first_rate) & reported[:, j], monthly_rate[:, j], first_rate)
    last_rate = monthly_rate[:, k - 1] if k else np.full(n, np.nan)
    trend = np.where(months_reported >= 2, last_rate - first_rate, np.nan)

    out = {
        prefix + "att_rate": _pooled_rate(att, wd),
        prefix + "days_attended": days.astype("float64"),
        prefix + "working_days": work.astype("float64"),
        prefix + "months_reported": months_reported.astype("float64"),
        prefix + "months_zero": months_zero.astype("float64"),
        prefix + "trailing_zero_months": trailing,
        prefix + "first_month_rate": first_rate,
        prefix + "last_month_rate": last_rate.astype("float64"),
        prefix + "trend": trend,
        prefix + "last_att_month": last_att,
        prefix + "any_attendance": (days > 0).astype("float64"),
    }
    for quarter, q_months in config.QUARTERS.items():
        idx = [i for i, m in enumerate(months) if m in q_months]
        if idx:
            out[prefix + quarter + "_rate"] = _pooled_rate(att[:, idx], wd[:, idx])
        else:
            out[prefix + quarter + "_rate"] = np.full(n, np.nan)
    return out
