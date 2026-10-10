"""Dropout label for base year T from evidence of study in year T+1.

The source system keeps students on the roll after they leave, so "a row exists next year"
is not evidence of study. The label uses attendance in the label window of T+1 (spec 3.1):

    evidence in window (person or student match)        -> 0 CONTINUED_ATTENDING
    next-year row in class >= 11                        -> 0 CONTINUED_CLASS_11_PLUS
    no next-year row                                    -> 1 DROPOUT_NOT_FOUND
    row(s) without evidence, school reports attendance  -> 1 DROPOUT_NOT_ATTENDING
    row(s) without evidence, school does not report     -> NaN UNKNOWN_SCHOOL_NOT_REPORTING
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from dropout_v2 import config


def school_reporting_share(df: pd.DataFrame, months: List[str]) -> pd.Series:
    """Share of each school's rows that have working days in `months` (index: school_key)."""
    valid = df["school_key"] >= 0
    if not months:
        return pd.Series(dtype="float64")
    has_data = df[["wd_{}".format(m) for m in months]].notna().any(axis=1)
    return has_data[valid].groupby(df.loc[valid, "school_key"]).mean()


def build_labels(base: pd.DataFrame, nxt: pd.DataFrame, window: Optional[List[str]] = None,
                 reporting_min: Optional[float] = None, class_continued_min: Optional[int] = None) -> pd.DataFrame:
    """Labels aligned row-by-row with `base` (needs person_key, student_key).

    `nxt` is the cleaned frame of year T+1 (one row per person_key).
    Returns columns person_key, label (1.0 / 0.0 / NaN), label_reason (category), with base's index.
    """
    window = list(window or config.LABEL_WINDOW)
    reporting_min = config.SCHOOL_REPORTING_MIN if reporting_min is None else reporting_min
    class_continued_min = config.CLASS_CONTINUED_MIN if class_continued_min is None else class_continued_min

    evidence = (nxt[["att_{}".format(m) for m in window]].fillna(0) > 0).any(axis=1)
    share = school_reporting_share(nxt, window)
    reporting = nxt["school_key"].map(share).fillna(0.0) >= reporting_min
    info = pd.DataFrame({
        "person_key": nxt["person_key"].to_numpy(),
        "student_key": nxt["student_key"].to_numpy(),
        "ev": evidence.to_numpy().astype("int8"),
        "c11": (nxt["class_t"] >= class_continued_min).to_numpy().astype("int8"),
        "rep": reporting.to_numpy().astype("int8"),
    })
    flags = ["ev", "c11", "rep"]
    by_person = info.groupby("person_key")[flags].max()
    by_student = info.loc[info["student_key"] >= 0].groupby("student_key")[flags].max()

    p = by_person.reindex(base["person_key"].to_numpy())
    student_keys = base["student_key"].to_numpy()
    s = by_student.reindex(np.where(student_keys >= 0, student_keys, -999))

    found_p = p["ev"].notna().to_numpy()
    found_s = s["ev"].notna().to_numpy()
    ev = (p["ev"].fillna(0).to_numpy() > 0) | (s["ev"].fillna(0).to_numpy() > 0)
    c11 = (p["c11"].fillna(0).to_numpy() > 0) | (s["c11"].fillna(0).to_numpy() > 0)
    rep = (p["rep"].fillna(0).to_numpy() > 0) | (s["rep"].fillna(0).to_numpy() > 0)
    found = found_p | found_s

    conditions = [ev, c11, ~found, rep]
    reasons = ["CONTINUED_ATTENDING", "CONTINUED_CLASS_11_PLUS", "DROPOUT_NOT_FOUND", "DROPOUT_NOT_ATTENDING"]
    reason = np.select(conditions, reasons, default="UNKNOWN_SCHOOL_NOT_REPORTING")
    label = np.select(conditions, [0.0, 0.0, 1.0, 1.0], default=np.nan)

    return pd.DataFrame({
        "person_key": base["person_key"].to_numpy(),
        "label": label.astype("float64"),
        "label_reason": pd.Categorical(reason, categories=config.LABEL_REASONS),
    }, index=base.index)
