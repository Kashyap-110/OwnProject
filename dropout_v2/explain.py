"""Per-student reasons from LightGBM contributions (exact TreeSHAP via pred_contrib) and global importance."""
from __future__ import annotations

from typing import Callable, Dict

import lightgbm as lgb
import numpy as np
import pandas as pd

from dropout_v2 import config, reference

# Features never cited as a reason: sensitive, not actionable, or only describing data availability.
# The model still uses them.
REASON_EXCLUDE = {"gender", "social_category", "district", "school_prev_n", "cur_working_days",
                  "cur_months_reported", "prev_months_reported", "school_cur_reporting_share"}

# At most one reason per topic, so three reasons say three different things.
_TOPICS = {
    "current attendance": ["cur_att_rate", "cur_trailing_zero_months", "cur_months_zero", "cur_last_month_rate",
                           "cur_first_month_rate", "cur_trend", "cur_days_attended", "cur_last_att_month",
                           "cur_q1_rate", "cur_q2_rate", "cur_q3_rate", "cur_q4_rate", "att_vs_school",
                           "engagement_status"],
    "previous attendance": ["prev_att_rate", "prev_last_q_att_rate", "prev_months_zero", "prev_last_att_month",
                            "prev_no_attendance"],
    "previous marks": ["prev_fa_rank", "prev_sa_rank", "prev_fa_zero", "prev_sa_zero", "prev_marks_missing"],
    "current marks": ["cur_fa_rank", "cur_sa_rank", "cur_fa_zero", "cur_sa_zero", "cur_marks_missing"],
    "age": ["age_t", "over_age", "dob_missing"],
    "school size": ["school_rows_t", "school_size_growth", "school_new_in_t"],
    "school history": ["prev_status", "rejoiner_flag", "prev_same_school", "prev_n_rows", "n_rows_in_year"],
    "parents": ["has_father_id", "has_mother_id"],
    "past dropout rates": ["school_prev_dropout_rate", "mandal_prev_dropout_rate"],
}
TOPIC_OF = {feature: topic for topic, members in _TOPICS.items() for feature in members}

# Plain wording when the value is missing.
MISSING_TEXT = {
    "cur_att_rate": "No attendance recorded this year",
    "cur_last_month_rate": "No attendance record for the latest month",
    "cur_first_month_rate": "No attendance record in the first month",
    "cur_trend": "Attendance recorded in fewer than two months this year",
    "cur_last_att_month": "Not attended on any day this year",
    "att_vs_school": "No attendance recorded this year",
    "school_cur_att_rate": "School has not sent attendance this year",
    "prev_att_rate": "No attendance record last year",
    "prev_last_q_att_rate": "No attendance record at the end of last year",
    "prev_last_att_month": "Not attended on any day last year",
    "prev_no_attendance": "Not in school records last year",
    "prev_same_school": "Not in school records last year",
    "rejoiner_flag": "Earlier school history not available",
    "prev_fa_rank": "No FA marks last year",
    "prev_sa_rank": "No SA marks last year",
    "cur_fa_rank": "No FA marks this year",
    "cur_sa_rank": "No SA marks this year",
    "age_t": "Date of birth missing or invalid",
    "over_age": "Date of birth missing or invalid",
    "school_prev_dropout_rate": "No dropout history for the school",
    "mandal_prev_dropout_rate": "No dropout history for the mandal",
    "school_size_growth": "School not in last year's records",
}


def _pct(v: float) -> str:
    return "{:.0%}".format(v)


def _num(v: float) -> str:
    return "{:,.0f}".format(v)


def _months(v: float) -> str:
    n = int(round(v))
    return "{} month{}".format(n, "" if n == 1 else "s")


QUARTER_TEXT = {"q1": "Jun-Aug", "q2": "Sep-Nov", "q3": "Dec-Feb", "q4": "Mar-Apr"}
ENGAGEMENT_TEXT = {
    "ATTENDING": "attending",
    "NOT_ATTENDING": "no attendance this year although the school reports attendance",
    "NO_SCHOOL_DATA": "the school has not sent attendance",
    "UNKNOWN": "not yet known",
}

# feature -> (label used when the value is missing, formatter for a present value)
_TEMPLATES: Dict[str, tuple] = {
    "cur_att_rate": ("Attendance this year so far", lambda v: "Attendance this year so far: " + _pct(v)),
    "cur_trailing_zero_months": ("Recent attendance", lambda v: "No attendance in the last " + _months(v)
                                 if v > 0 else "Attended in the latest month"),
    "cur_months_zero": ("Zero-attendance months", lambda v: _months(v) + " with zero attendance this year"),
    "cur_last_month_rate": ("Attendance last month", lambda v: "Attendance last month: " + _pct(v)),
    "cur_first_month_rate": ("Attendance in the first month", lambda v: "Attendance in the first month: " + _pct(v)),
    "cur_trend": ("Attendance trend", lambda v: "Attendance changed by {:+.0f} points this year".format(v * 100)),
    "cur_days_attended": ("Days attended", lambda v: "Days attended this year: " + _num(v)),
    "cur_working_days": ("Working days recorded", lambda v: "Working days recorded this year: " + _num(v)),
    "cur_months_reported": ("Months with attendance records", lambda v: "Months with attendance records: " + _num(v)),
    "cur_last_att_month": ("Last month attended", lambda v: "Last attended in month {} of the year".format(int(v))),
    "att_vs_school": ("Attendance compared with school",
                      lambda v: "Attendance {:+.0f} points compared with the school average".format(v * 100)),
    "school_cur_att_rate": ("School attendance", lambda v: "School's average attendance this year: " + _pct(v)),
    "school_cur_reporting_share": ("School attendance records",
                                   lambda v: "Students of the school with attendance records: " + _pct(v)),
    "prev_att_rate": ("Attendance last year", lambda v: "Attendance last year: " + _pct(v)),
    "prev_last_q_att_rate": ("Attendance at the end of last year",
                             lambda v: "Attendance at the end of last year (Mar-Apr): " + _pct(v)),
    "prev_months_zero": ("Zero-attendance months last year", lambda v: _months(v) + " with zero attendance last year"),
    "prev_months_reported": ("Attendance records last year", lambda v: "Months with attendance records last year: " + _num(v)),
    "prev_last_att_month": ("Last month attended last year",
                            lambda v: "Last attended in month {} of last year".format(int(v))),
    "prev_no_attendance": ("Attendance last year", lambda v: "No attendance at all last year" if v >= 1
                           else "Attended last year"),
    "prev_same_school": ("School change", lambda v: "Same school as last year" if v >= 1
                         else "Changed school since last year"),
    "rejoiner_flag": ("Gap year", lambda v: "Returned after a year out of school records" if v >= 1
                      else "No gap year in school records"),
    "prev_fa_rank": ("Marks last year", lambda v: "FA marks last year: percentile {:.0f}".format(v * 100)),
    "prev_sa_rank": ("Marks last year", lambda v: "SA marks last year: percentile {:.0f}".format(v * 100)),
    "prev_fa_zero": ("Marks last year", lambda v: "Zero FA marks last year" if v >= 1 else "FA marks recorded last year"),
    "prev_sa_zero": ("Marks last year", lambda v: "Zero SA marks last year" if v >= 1 else "SA marks recorded last year"),
    "prev_marks_missing": ("Marks last year", lambda v: "No marks recorded last year" if v >= 1
                           else "Marks recorded last year"),
    "prev_n_rows": ("Records last year", lambda v: "Enrolled in {} schools last year".format(int(v))),
    "cur_fa_rank": ("Marks this year", lambda v: "FA marks this year: percentile {:.0f}".format(v * 100)),
    "cur_sa_rank": ("Marks this year", lambda v: "SA marks this year: percentile {:.0f}".format(v * 100)),
    "cur_fa_zero": ("Marks this year", lambda v: "Zero FA marks this year" if v >= 1 else "FA marks recorded this year"),
    "cur_sa_zero": ("Marks this year", lambda v: "Zero SA marks this year" if v >= 1 else "SA marks recorded this year"),
    "cur_marks_missing": ("Marks this year", lambda v: "No marks recorded this year" if v >= 1
                          else "Marks recorded this year"),
    "n_rows_in_year": ("Records this year", lambda v: "Enrolled in {} schools this year".format(int(v))),
    "age_t": ("Age", lambda v: "Age {}".format(int(v))),
    "over_age": ("Age for class", lambda v: "Older than usual for the class by {} year{}".format(int(v), "" if int(v) == 1 else "s")
                 if v > 0 else "Age normal for the class"),
    "dob_missing": ("Date of birth", lambda v: "Date of birth missing or invalid" if v >= 1 else "Date of birth recorded"),
    "class_t": ("Class", lambda v: "Class {}".format(int(v))),
    "is_terminal_class": ("School structure", lambda v: "Last class offered by the school: must change school next year"
                          if v >= 1 else "School continues to the next class"),
    "home_school_km": ("Distance from home to school", lambda v: "Lives {:.1f} km from school".format(v)),
    "school_rows_t": ("School enrolment", lambda v: "School enrolment: {} students".format(_num(v))),
    "school_size_growth": ("School enrolment change", lambda v: "School enrolment is {} of last year".format(_pct(v))),
    "school_new_in_t": ("School history", lambda v: "School is new in the records this year" if v >= 1
                        else "School existed last year"),
    "ptr": ("Pupil-teacher ratio", lambda v: "Pupil-teacher ratio {:.0f}".format(v)),
    "school_prev_dropout_rate": ("School dropout rate last year",
                                 lambda v: "School dropout rate last year: {:.1%}".format(v)),
    "school_prev_n": ("School size last year", lambda v: "School had {} students last year".format(_num(v))),
    "mandal_prev_dropout_rate": ("Mandal dropout rate last year",
                                 lambda v: "Mandal dropout rate last year: {:.1%}".format(v)),
    "has_father_id": ("Father record", lambda v: "Father linked in household records" if v >= 1
                      else "Father not linked in household records"),
    "has_mother_id": ("Mother record", lambda v: "Mother linked in household records" if v >= 1
                      else "Mother not linked in household records"),
    "is_tribal_mandal": ("Tribal mandal", lambda v: "Lives in a tribal mandal" if v >= 1 else "Not a tribal mandal"),
}
for _q, _text in QUARTER_TEXT.items():
    _TEMPLATES["cur_{}_rate".format(_q)] = (
        "Attendance in {}".format(_text), (lambda t: (lambda v: "Attendance in {}: {}".format(t, _pct(v))))(_text))
    MISSING_TEXT["cur_{}_rate".format(_q)] = "No attendance record in {}".format(_text)

_CATEGORY_LABELS = {
    "engagement_status": ("Attendance status", lambda c: "Attendance status: " + ENGAGEMENT_TEXT.get(c, c.lower())),
    "prev_status": ("Last year", lambda c: "Last year: " + reference.PREV_STATUS_TEXT.get(c, c.lower())),
    "mgmt_group": ("School management", lambda c: "School management: " + c.replace("_", " ").lower()),
    "school_category": ("School category", lambda c: "School category: " + c.lower()),
    "board_group": ("Board", lambda c: "Board: " + c),
    "school_gender_type": ("School type", lambda c: "School type: " + c.replace("_", "-").lower()),
    "school_rural_urban": ("School location", lambda c: "School location: " + c.lower()),
    "rural_urban": ("Home location", lambda c: "Home location: " + c.lower()),
    "ration_card": ("Ration card", lambda c: "Ration card: " + c),
    "dwelling": ("Dwelling", lambda c: "Dwelling: " + c.lower()),
    "medium": ("Medium of instruction", lambda c: "Medium of instruction: " + c.lower()),
}


def _format_values(feature: str, values: pd.Series) -> np.ndarray:
    if feature in _CATEGORY_LABELS:
        label, fmt = _CATEGORY_LABELS[feature]
        text = values.astype(object).where(values.notna(), config.UNKNOWN).astype(str)
        return np.array([label + ": not recorded" if t == config.UNKNOWN else fmt(t) for t in text], dtype=object)
    if isinstance(values.dtype, pd.CategoricalDtype) or values.dtype == object:
        text = values.astype(object).where(values.notna(), config.UNKNOWN).astype(str)
        name = feature.replace("_", " ").capitalize()
        return np.array([name + ": " + t.lower() for t in text], dtype=object)
    label, fmt = _TEMPLATES.get(feature, (feature.replace("_", " ").capitalize(), None))
    missing_text = MISSING_TEXT.get(feature, label + ": not recorded")
    nums = pd.to_numeric(values, errors="coerce").to_numpy(dtype="float64")
    out = np.empty(len(nums), dtype=object)
    for i, v in enumerate(nums):
        if np.isnan(v):
            out[i] = missing_text
        elif fmt is None:
            out[i] = "{}: {:.3g}".format(label, v)
        else:
            out[i] = fmt(v)
    return out


def contributions(booster: lgb.Booster, X: pd.DataFrame, num_iteration: int, chunk: int = 500_000) -> np.ndarray:
    """Per-row, per-feature contributions to the raw score (bias column dropped). float32."""
    parts = []
    for start in range(0, len(X), chunk):
        c = booster.predict(X.iloc[start:start + chunk], num_iteration=num_iteration, pred_contrib=True)
        parts.append(np.asarray(c)[:, :-1].astype("float32"))
    if not parts:
        return np.zeros((0, X.shape[1]), dtype="float32")
    return np.vstack(parts)


def top_reasons(contrib: np.ndarray, values: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    """Plain-language reasons: the k topics with the largest positive contribution, one feature per topic."""
    names = list(values.columns)
    n = len(values)
    usable = np.array([name not in REASON_EXCLUDE for name in names])
    scores = np.where(usable[None, :] & (contrib > 0), contrib, -np.inf)

    topics: dict = {}
    for idx, name in enumerate(names):
        topics.setdefault(TOPIC_OF.get(name, name), []).append(idx)
    topic_cols = list(topics.values())
    topic_score = np.full((n, len(topic_cols)), -np.inf)
    topic_feature = np.zeros((n, len(topic_cols)), dtype=int)
    for t, cols in enumerate(topic_cols):
        block = scores[:, cols]
        best = np.argmax(block, axis=1) if n else np.zeros(0, dtype=int)
        topic_score[:, t] = block[np.arange(n), best] if n else np.zeros(0)
        topic_feature[:, t] = np.asarray(cols)[best] if n else np.zeros(0, dtype=int)
    order = np.argsort(-topic_score, axis=1, kind="mergesort")

    out = pd.DataFrame(index=values.index)
    for slot in range(k):
        text = np.full(n, "", dtype=object)
        if slot < order.shape[1]:
            chosen_topic = order[:, slot]
            chosen_score = topic_score[np.arange(n), chosen_topic]
            feature_idx = topic_feature[np.arange(n), chosen_topic]
            for f in np.unique(feature_idx):
                rows = np.where((feature_idx == f) & np.isfinite(chosen_score))[0]
                if len(rows):
                    text[rows] = _format_values(names[f], values.iloc[rows, f])
        out["reason_{}".format(slot + 1)] = text
    return out


def global_importance(booster: lgb.Booster, X: pd.DataFrame, num_iteration: int, max_rows: int = 200_000) -> pd.DataFrame:
    sample = X if len(X) <= max_rows else X.sample(n=max_rows, random_state=config.RANDOM_SEED)
    contrib = contributions(booster, sample, num_iteration)
    gain = booster.feature_importance(importance_type="gain", iteration=num_iteration)
    imp = pd.DataFrame({
        "feature": list(X.columns),
        "mean_abs_contribution": np.abs(contrib).mean(axis=0) if len(contrib) else 0.0,
        "gain": gain,
    })
    return imp.sort_values("mean_abs_contribution", ascending=False).reset_index(drop=True)
