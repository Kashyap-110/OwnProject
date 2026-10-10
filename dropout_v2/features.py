"""Time-safe features for base year T and one snapshot (spec section 5).

A snapshot feature uses only: profile columns, months of T up to the snapshot, years T-1 and
T-2, and labels of base year T-1 (from the `jul` snapshot onward). tests/test_features.py
changes all later data and checks that the features do not move.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from dropout_v2 import config, reference
from dropout_v2.attendance import attendance_features, month_arrays
from dropout_v2.clean import clean_text
from dropout_v2.labels import school_reporting_share

ENGAGEMENT_LEVELS = ["ATTENDING", "NOT_ATTENDING", "NO_SCHOOL_DATA", "UNKNOWN"]
PREV_STATUS_LEVELS = ["PROMOTED", "REPEATED", "OTHER_CHANGE", "NOT_IN_PREV_YEAR",
                      "PREV_YEAR_DATA_GAP", "PREV_YEAR_NOT_AVAILABLE"]

ID_COLUMNS = ["person_key", "person_id", "student_key", "school_key", "academic_year", "district", "mandal",
              "class_t", "gender", "social_category", "mgmt_group", "engagement_status"]

_PROFILE_NUMERIC = ["class_t", "age_t", "over_age", "dob_missing", "has_father_id", "has_mother_id",
                    "is_tribal_mandal", "home_school_km"]
_PROFILE_CATEGORICAL = ["gender", "rural_urban", "ration_card", "dwelling", "medium", "district"]
_SCHOOL_NUMERIC = ["school_rows_t", "school_size_growth", "school_new_in_t", "ptr", "is_terminal_class"]
_SCHOOL_CATEGORICAL = ["mgmt_group", "school_category", "board_group", "school_gender_type", "school_rural_urban"]
_PREV_NUMERIC = ["prev_same_school", "rejoiner_flag", "prev_att_rate", "prev_last_q_att_rate",
                 "prev_months_reported", "prev_months_zero", "prev_last_att_month", "prev_no_attendance",
                 "prev_n_rows", "prev_fa_rank", "prev_sa_rank", "prev_fa_zero", "prev_sa_zero", "prev_marks_missing"]
_CURRENT_NUMERIC = ["cur_att_rate", "cur_days_attended", "cur_working_days", "cur_months_reported",
                    "cur_months_zero", "cur_trailing_zero_months", "cur_first_month_rate", "cur_last_month_rate",
                    "cur_trend", "cur_last_att_month", "school_cur_reporting_share", "school_cur_att_rate",
                    "att_vs_school"]
_RATE_NUMERIC = ["school_prev_dropout_rate", "school_prev_n", "mandal_prev_dropout_rate"]
_END_OF_YEAR_NUMERIC = ["cur_fa_rank", "cur_sa_rank", "cur_fa_zero", "cur_sa_zero", "cur_marks_missing",
                        "n_rows_in_year"]


def feature_columns(snapshot: str) -> Tuple[List[str], List[str]]:
    """(numeric, categorical) model inputs for a snapshot. The single source of feature names."""
    months = config.visible_months(snapshot)
    numeric = list(_PROFILE_NUMERIC) + list(_SCHOOL_NUMERIC) + list(_PREV_NUMERIC)
    categorical = list(_PROFILE_CATEGORICAL) + list(_SCHOOL_CATEGORICAL) + ["prev_status"]
    if config.USE_SOCIAL_CATEGORY:
        categorical.append("social_category")
    if months:
        numeric += _CURRENT_NUMERIC
        numeric += ["cur_{}_rate".format(q) for q, q_months in config.QUARTERS.items()
                    if any(m in months for m in q_months)]
        numeric += _RATE_NUMERIC
        categorical.append("engagement_status")
    if snapshot in config.END_OF_YEAR_SNAPSHOTS:
        numeric += _END_OF_YEAR_NUMERIC
    return numeric, categorical


def prepare_school_dim(schools: pd.DataFrame) -> pd.DataFrame:
    """School dimension indexed by school_key with cleaned attributes."""
    from dropout_v2.clean import to_key

    dim = pd.DataFrame(index=schools.index)
    dim["school_key"] = to_key(schools["school_id"])
    dim["school_name"] = schools["school_name"].astype("string")
    mgmt = clean_text(schools["school_management"])
    dim["mgmt_group"] = pd.Categorical(
        [config.UNKNOWN if m == config.UNKNOWN else reference.MANAGEMENT_GROUPS.get(m, config.OTHER)
         for m in mgmt.astype(str)])
    dim["school_category"] = clean_text(schools["school_category"])
    dim["board_group"] = clean_text(schools["board_affiliation"], {"STATE": "STATE", "CBSE": "CBSE", "ICSE": "ICSE"},
                                    config.OTHER)
    dim["school_gender_type"] = clean_text(schools["school_type_coed_boys_girls"],
                                           {"CO_ED": "CO_ED", "CO-ED": "CO_ED", "COED": "CO_ED",
                                            "GIRLS": "GIRLS", "BOYS": "BOYS"}, config.OTHER)
    dim["school_rural_urban"] = clean_text(schools["school_in_rural_urban"],
                                           {"RURAL": "RURAL", "R": "RURAL", "URBAN": "URBAN", "U": "URBAN"},
                                           config.OTHER)
    for col in ("total_students", "total_teachers", "lowest_class", "highest_class"):
        dim[col] = pd.to_numeric(schools[col], errors="coerce").astype("float64")
    dim["total_teachers"] = dim["total_teachers"].where(dim["total_teachers"] > 0)
    lat = pd.to_numeric(schools["school_latitude"], errors="coerce")
    lon = pd.to_numeric(schools["school_longitude"], errors="coerce")
    ok = lat.between(*config.LAT_RANGE) & lon.between(*config.LON_RANGE)
    dim["school_lat"] = lat.where(ok)
    dim["school_lon"] = lon.where(ok)
    dim = dim[dim["school_key"] >= 0].drop_duplicates("school_key").set_index("school_key")
    return dim


def base_population(frame: pd.DataFrame) -> pd.DataFrame:
    """Rows that get a risk score: classes 1-9 and not deceased (index reset)."""
    keep = frame["class_t"].isin(config.BASE_CLASSES) & ~frame["deceased"]
    return frame[keep].reset_index(drop=True)


def _haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(x, dtype="float64")) for x in (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2.0) ** 2
    return 6371.0 * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _as_category(values, levels: Optional[List[str]] = None) -> pd.Categorical:
    if levels is not None:
        return pd.Categorical(values, categories=levels)
    return pd.Categorical(values)


def _school_pooled_rate(frame: pd.DataFrame, months: List[str]) -> pd.Series:
    valid = frame["school_key"] >= 0
    att = frame.loc[valid, ["att_{}".format(m) for m in months]].sum(axis=1, min_count=1)
    wd = frame.loc[valid, ["wd_{}".format(m) for m in months]].sum(axis=1, min_count=1)
    grouped = pd.DataFrame({"att": att, "wd": wd, "school_key": frame.loc[valid, "school_key"]}).groupby("school_key")
    sums = grouped[["att", "wd"]].sum(min_count=1)
    return (sums["att"] / sums["wd"].where(sums["wd"] > 0)).astype("float64")


def _smoothed_rate(keys: pd.Series, label: pd.Series, weight: float) -> Tuple[pd.Series, pd.Series, float]:
    ok = label.notna()
    overall = float(label[ok].mean()) if ok.any() else np.nan
    grouped = label[ok].groupby(keys[ok])
    total = grouped.sum()
    count = grouped.size()
    rate = (total + weight * overall) / (count + weight)
    return rate, count, overall


def build_features(year: str, snapshot: str, frames: Dict[str, pd.DataFrame], schools_dim: pd.DataFrame,
                   prev_labels: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Feature frame for base year `year` (classes 1-9, not deceased) at `snapshot`.

    frames: academic year -> cleaned frame (years missing from the data are absent).
    prev_labels: labels of base year T-1 with columns label, school_key, mandal_key (or None).
    """
    if year not in frames:
        raise KeyError("Cleaned data for {} is missing".format(year))
    months = config.visible_months(snapshot)
    cur = frames[year]
    base = base_population(cur)
    out = pd.DataFrame(index=base.index)
    for col in ["person_key", "person_id", "student_key", "school_key", "academic_year", "district", "mandal",
                "gender", "social_category", "rural_urban", "ration_card", "dwelling", "medium"]:
        out[col] = base[col]
    out["class_t"] = base["class_t"].astype("float64")
    _profile(out, base, year)
    _school(out, base, frames, year, schools_dim)
    _previous_year(out, base, frames, year)
    _current_year(out, base, cur, months)
    if months:
        _rates(out, prev_labels)
    if snapshot in config.END_OF_YEAR_SNAPSHOTS:
        out["cur_fa_rank"] = base["fa_rank"].astype("float64")
        out["cur_sa_rank"] = base["sa_rank"].astype("float64")
        out["cur_fa_zero"] = base["fa_zero"].astype("float64")
        out["cur_sa_zero"] = base["sa_zero"].astype("float64")
        out["cur_marks_missing"] = base["marks_missing"].astype("float64")
        out["n_rows_in_year"] = base["n_rows_in_year"].astype("float64")
    out["snapshot"] = snapshot
    # float32 halves memory on ~7M rows per year; the model uses float32 anyway.
    wide = [c for c in out.columns if out[c].dtype == np.float64]
    out[wide] = out[wide].astype("float32")
    return out


def _profile(out: pd.DataFrame, base: pd.DataFrame, year: str) -> None:
    june1 = pd.Timestamp(config.year_start(year), 6, 1)
    days = (june1 - base["dob"]).dt.days.astype("float64")
    age = np.floor(days / 365.25)
    valid = (age >= config.AGE_MIN) & (age <= config.AGE_MAX)
    out["age_t"] = age.where(valid)
    out["over_age"] = out["age_t"] - (out["class_t"] + 5.0)
    out["dob_missing"] = (~valid).astype("float64")
    out["has_father_id"] = base["has_father_id"].astype("float64")
    out["has_mother_id"] = base["has_mother_id"].astype("float64")
    tribal = {"{}|{}".format(d, m) for d, m in reference.TRIBAL_MANDALS}
    key = base["district"].astype(str) + "|" + base["mandal"].astype(str)
    out["is_tribal_mandal"] = key.isin(tribal).astype("float64")


def _school(out: pd.DataFrame, base: pd.DataFrame, frames: Dict[str, pd.DataFrame], year: str,
            dim: pd.DataFrame) -> None:
    joined = dim.reindex(base["school_key"].to_numpy())
    joined.index = base.index
    for col in ["mgmt_group", "school_category", "board_group", "school_gender_type", "school_rural_urban"]:
        values = joined[col].astype(object).where(joined[col].notna(), config.UNKNOWN)
        out[col] = _as_category(values)
    teachers = joined["total_teachers"]
    out["ptr"] = (joined["total_students"] / teachers).where(teachers > 0)
    highest = joined["highest_class"]
    out["is_terminal_class"] = np.where(highest.notna(), (out["class_t"] == highest).astype("float64"), np.nan)
    km = _haversine_km(base["house_lat"], base["house_lon"], joined["school_lat"], joined["school_lon"])
    out["home_school_km"] = np.where(km <= config.MAX_HOME_SCHOOL_KM, km, np.nan)

    cur = frames[year]
    rows_t = cur.loc[cur["school_key"] >= 0].groupby("school_key").size()
    out["school_rows_t"] = base["school_key"].map(rows_t).astype("float64")
    prev = config.prev_year(year)
    if prev in frames:
        p = frames[prev]
        rows_prev = p.loc[p["school_key"] >= 0].groupby("school_key").size()
        prev_count = base["school_key"].map(rows_prev).astype("float64")
        out["school_size_growth"] = out["school_rows_t"] / prev_count
        out["school_new_in_t"] = prev_count.isna().astype("float64")
    else:
        out["school_size_growth"] = np.nan
        out["school_new_in_t"] = np.nan


def _previous_year(out: pd.DataFrame, base: pd.DataFrame, frames: Dict[str, pd.DataFrame], year: str) -> None:
    prev = config.prev_year(year)
    numeric_prev = [c for c in _PREV_NUMERIC]
    if prev not in frames:
        for col in numeric_prev:
            out[col] = np.nan
        out["prev_status"] = _as_category(["PREV_YEAR_NOT_AVAILABLE"] * len(out), PREV_STATUS_LEVELS)
        return
    cols = ["person_key", "class_t", "school_key", "fa_rank", "sa_rank", "fa_zero", "sa_zero", "marks_missing",
            "n_rows_in_year"] + ["att_{}".format(m) for m in config.MONTHS] + ["wd_{}".format(m) for m in config.MONTHS]
    p = frames[prev][cols].set_index("person_key").reindex(base["person_key"].to_numpy())
    p.index = base.index
    found = p["class_t"].notna().to_numpy()
    diff = out["class_t"].to_numpy() - p["class_t"].to_numpy(dtype="float64")
    gap_classes = config.KNOWN_GAPS.get(prev, [])
    expected_prev = np.where(out["class_t"].to_numpy() == 1, -1, out["class_t"].to_numpy() - 1)
    in_gap = np.isin(expected_prev, gap_classes)
    status = np.select(
        [found & (diff == 1), found & (diff == 0), found, in_gap],
        ["PROMOTED", "REPEATED", "OTHER_CHANGE", "PREV_YEAR_DATA_GAP"],
        default="NOT_IN_PREV_YEAR")
    out["prev_status"] = _as_category(status, PREV_STATUS_LEVELS)
    out["prev_same_school"] = np.where(found, (p["school_key"].to_numpy() == base["school_key"].to_numpy()).astype(float),
                                       np.nan)
    att, wd = month_arrays(p, config.MONTHS)
    feats = attendance_features(att, wd, config.MONTHS, "prev_")
    last_q = [config.MONTHS.index(m) for m in config.PREV_YEAR_LAST_QUARTER]
    q_att, q_wd = att[:, last_q], wd[:, last_q]
    with np.errstate(invalid="ignore", divide="ignore"):
        q_work = np.nansum(q_wd, axis=1)
        q_rate = np.where(q_work > 0, np.nansum(q_att, axis=1) / np.where(q_work > 0, q_work, 1.0), np.nan)
    mask = lambda values: np.where(found, values, np.nan)  # noqa: E731 - values only for students found in T-1
    out["prev_att_rate"] = mask(feats["prev_att_rate"])
    out["prev_last_q_att_rate"] = mask(q_rate)
    out["prev_months_reported"] = mask(feats["prev_months_reported"])
    out["prev_months_zero"] = mask(feats["prev_months_zero"])
    out["prev_last_att_month"] = mask(feats["prev_last_att_month"])
    out["prev_no_attendance"] = mask(1.0 - feats["prev_any_attendance"])
    out["prev_n_rows"] = p["n_rows_in_year"].astype("float64").to_numpy()
    for col in ("fa_rank", "sa_rank", "fa_zero", "sa_zero", "marks_missing"):
        out["prev_" + col] = p[col].astype("float64").to_numpy()

    prev2 = config.prev_year(year, 2)
    if prev2 in frames:
        in_prev2 = base["person_key"].isin(frames[prev2]["person_key"]).to_numpy()
        out["rejoiner_flag"] = (~found & in_prev2).astype("float64")
    else:
        out["rejoiner_flag"] = np.nan


def _current_year(out: pd.DataFrame, base: pd.DataFrame, cur: pd.DataFrame, months: List[str]) -> None:
    if not months:
        out["engagement_status"] = _as_category(["UNKNOWN"] * len(out), ENGAGEMENT_LEVELS)
        return
    att, wd = month_arrays(base, months)
    feats = attendance_features(att, wd, months, "cur_")
    for key, values in feats.items():
        if key == "cur_any_attendance":
            continue
        out[key] = values
    share = school_reporting_share(cur, months)
    out["school_cur_reporting_share"] = base["school_key"].map(share).astype("float64").to_numpy()
    school_rate = _school_pooled_rate(cur, months)
    out["school_cur_att_rate"] = base["school_key"].map(school_rate).astype("float64").to_numpy()
    out["att_vs_school"] = out["cur_att_rate"] - out["school_cur_att_rate"]
    # Engagement is judged on the latest visible month: "not seen this month" is what field staff act on.
    latest = months[-1]
    attended_latest = base["att_{}".format(latest)].fillna(0).to_numpy() > 0
    latest_share = base["school_key"].map(school_reporting_share(cur, [latest])).fillna(0.0).to_numpy()
    status = np.where(attended_latest, "ATTENDING",
                      np.where(latest_share >= config.SCHOOL_REPORTING_MIN, "NOT_ATTENDING", "NO_SCHOOL_DATA"))
    out["engagement_status"] = _as_category(status, ENGAGEMENT_LEVELS)


def _rates(out: pd.DataFrame, prev_labels: Optional[pd.DataFrame]) -> None:
    if prev_labels is None or len(prev_labels) == 0:
        for col in _RATE_NUMERIC:
            out[col] = np.nan
        return
    w = config.RATE_SMOOTHING_WEIGHT
    school_rate, school_n, _ = _smoothed_rate(prev_labels["school_key"], prev_labels["label"], w)
    out["school_prev_dropout_rate"] = out["school_key"].map(school_rate).astype("float64")
    out["school_prev_n"] = out["school_key"].map(school_n).astype("float64")
    mandal_rate, _, _ = _smoothed_rate(prev_labels["mandal_key"], prev_labels["label"], w)
    key = out["district"].astype(str) + "|" + out["mandal"].astype(str)
    out["mandal_prev_dropout_rate"] = key.map(mandal_rate).astype("float64")
