"""Raw yearly extract -> cleaned frame with exactly one row per student (person_key).

Every rule here comes from the data audit (docs/data_team/staging_table_change_request.md).
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from dropout_v2 import config

ATT_COLS = ["att_{}".format(m) for m in config.MONTHS]
WD_COLS = ["wd_{}".format(m) for m in config.MONTHS]

TEXT_MAPS: Dict[str, Dict[str, str]] = {
    "student_gender": {
        "MALE": "MALE", "M": "MALE", "BOY": "MALE",
        "FEMALE": "FEMALE", "F": "FEMALE", "GIRL": "FEMALE",
        "TRANSGENDER": "OTHER", "OTHERS": "OTHER", "OTHER": "OTHER",
    },
    "rural_urban_flag": {"R": "RURAL", "RURAL": "RURAL", "U": "URBAN", "URBAN": "URBAN"},
    "ration_card_type": {"AAY": "AAY", "BPL": "BPL"},
    "social_category": {"SC": "SC", "ST": "ST", "BC": "BC", "OC": "OC"},
}
TEXT_DEFAULTS: Dict[str, Optional[str]] = {
    "student_gender": config.OTHER,
    "rural_urban_flag": config.OTHER,
    "ration_card_type": config.OTHER,
    "social_category": config.OTHER,
}

CLEAN_COLUMNS = (
    ["person_key", "person_id", "student_key", "school_key", "academic_year", "class_t", "dob",
     "gender", "social_category", "rural_urban", "ration_card", "dwelling", "medium", "district", "mandal",
     "has_father_id", "has_mother_id", "deceased", "house_lat", "house_lon", "src_is_dropout"]
    + ATT_COLS + WD_COLS
    + ["fa_marks", "sa_marks", "fa_rank", "sa_rank", "fa_zero", "sa_zero", "marks_missing",
       "n_rows_in_year", "n_schools_in_year"]
)


def to_key(values: pd.Series) -> pd.Series:
    """Digits-only ID text -> int64; anything else -> -1. Leading zeros do not matter for joins."""
    text = values.astype("string").str.strip()
    valid = text.str.fullmatch(r"\d{1,18}").fillna(False).astype(bool)
    numbers = pd.to_numeric(text.where(valid), errors="coerce")
    return numbers.fillna(-1).astype("int64")


def _clean_strings(values: pd.Series, mapping: Optional[Dict[str, str]], default: Optional[str]) -> pd.Series:
    text = values.astype("string").str.strip().str.upper().str.replace(r"\s+", " ", regex=True)
    text = text.fillna(config.UNKNOWN)
    text = text.where(~text.isin(config.NULL_TOKENS), config.UNKNOWN)
    if mapping is not None:
        mapped = text.map(mapping)
        if default is not None:
            mapped = mapped.where(mapped.notna() | (text == config.UNKNOWN), default)
        else:
            mapped = mapped.fillna(text)
        text = mapped.fillna(config.UNKNOWN)
    return text.astype(object)


def clean_text(values: pd.Series, mapping: Optional[Dict[str, str]] = None,
               default: Optional[str] = None) -> pd.Series:
    """Trim, upper-case, collapse spaces, map NULL tokens to UNKNOWN, optional code map. Returns category."""
    cat = values if isinstance(values.dtype, pd.CategoricalDtype) else values.astype("category")
    categories = pd.Series(cat.cat.categories.astype(str), dtype=object)
    cleaned = _clean_strings(categories, mapping, default).tolist() + [config.UNKNOWN]
    codes = cat.cat.codes.to_numpy()          # -1 for missing -> last entry (UNKNOWN)
    lookup = np.array(cleaned, dtype=object)
    out = lookup[codes]
    return pd.Series(pd.Categorical(out), index=values.index)


def _numeric(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").astype("float64")


def _clean_attendance(raw: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for month in config.MONTHS:
        att = _numeric(raw["{}_attended".format(month)])
        wd = _numeric(raw["{}_working_days".format(month)])
        wd = wd.where((wd > 0) & (wd <= config.MAX_WORKING_DAYS))     # 0 = no data; > 31 = invalid
        att = att.where(wd.notna() & att.notna())
        att = att.clip(lower=0)
        att = att.where(~(att > wd), wd)
        wd = wd.where(att.notna())                                     # attended missing -> month has no data
        out["att_{}".format(month)] = att
        out["wd_{}".format(month)] = wd
    return pd.DataFrame(out, index=raw.index)


def _marks_features(df: pd.DataFrame) -> pd.DataFrame:
    for name in ("fa", "sa"):
        col = "{}_marks".format(name)
        marks = df[col]
        positive = marks.where(marks > 0)
        p99 = positive.groupby(df["class_t"]).transform(lambda s: s.quantile(0.99))
        too_big = positive > config.MARKS_OUTLIER_FACTOR * p99
        marks = marks.where(~too_big)
        df[col] = marks
        df["{}_zero".format(name)] = (marks == 0).astype("int8")
        positive = marks.where(marks > 0)
        df["{}_rank".format(name)] = positive.groupby(df["class_t"]).rank(pct=True)
    df["marks_missing"] = (df["fa_marks"].isna() & df["sa_marks"].isna()).astype("int8")
    return df


def clean_year(raw: pd.DataFrame, year: str) -> Tuple[pd.DataFrame, dict]:
    """Clean one academic year. Returns (frame with CLEAN_COLUMNS, diagnostics)."""
    diag: dict = {"year": year, "rows_in": int(len(raw))}
    df = pd.DataFrame(index=raw.index)
    df["person_key"] = to_key(raw["person_id"])
    df["person_id"] = raw["person_id"].astype("string").str.strip()
    bad_person = df["person_key"] < 0
    diag["rows_bad_person_id"] = int(bad_person.sum())

    cls = pd.to_numeric(raw["class_currently_enrolled"], errors="coerce")
    bad_class = cls.isna() | (cls < config.KEEP_CLASS_MIN) | (cls > config.KEEP_CLASS_MAX) | (cls == 0)
    diag["rows_bad_class"] = int((bad_class & ~bad_person).sum())
    keep = ~bad_person & ~bad_class
    raw = raw.loc[keep]
    df = df.loc[keep].copy()
    df["class_t"] = cls.loc[keep].astype("int8")

    counts = df["person_key"].value_counts()
    placeholders = counts[counts > config.MAX_ROWS_PER_PERSON_YEAR]
    is_placeholder = df["person_key"].isin(placeholders.index)
    id_text = df.loc[is_placeholder].groupby("person_key")["person_id"].first()
    diag["placeholder_ids"] = {str(id_text.get(k, k)): int(v) for k, v in placeholders.items()}
    diag["rows_placeholder"] = int(is_placeholder.sum())
    raw = raw.loc[~is_placeholder]
    df = df.loc[~is_placeholder].copy()

    df["student_key"] = to_key(raw["student_id"])
    df["school_key"] = to_key(raw["school_id"])
    df["academic_year"] = year
    df = pd.concat([df, _clean_attendance(raw)], axis=1)
    df["fa_marks"] = _numeric(raw["fa_marks"])
    df["sa_marks"] = _numeric(raw["sa_marks"])
    df["dob"] = pd.to_datetime(raw["student_date_of_birth"].astype("string"), errors="coerce", format="ISO8601")
    df["gender"] = clean_text(raw["student_gender"], TEXT_MAPS["student_gender"], TEXT_DEFAULTS["student_gender"])
    df["social_category"] = clean_text(raw["social_category"], TEXT_MAPS["social_category"],
                                       TEXT_DEFAULTS["social_category"])
    df["rural_urban"] = clean_text(raw["rural_urban_flag"], TEXT_MAPS["rural_urban_flag"],
                                   TEXT_DEFAULTS["rural_urban_flag"])
    df["ration_card"] = clean_text(raw["ration_card_type"], TEXT_MAPS["ration_card_type"],
                                   TEXT_DEFAULTS["ration_card_type"])
    df["dwelling"] = clean_text(raw["dwelling_type"])
    df["medium"] = clean_text(raw["medium_name"])
    df["district"] = clean_text(raw["student_district_name"])
    df["mandal"] = clean_text(raw["student_mandal_name"])
    df["src_is_dropout"] = clean_text(raw["src_is_dropout"])
    deceased_text = clean_text(raw["is_deceased"])
    df["deceased"] = deceased_text.astype(str).isin(config.DECEASED_VALUES)
    for col in ("has_father_id", "has_mother_id"):
        df[col] = pd.to_numeric(raw[col], errors="coerce").fillna(0).clip(0, 1).astype("int8")
    lat = _numeric(raw["house_latitude"])
    lon = _numeric(raw["house_longitude"])
    ok = lat.between(*config.LAT_RANGE) & lon.between(*config.LON_RANGE)
    df["house_lat"] = lat.where(ok)
    df["house_lon"] = lon.where(ok)

    df = _deduplicate(df, diag)
    df = _marks_features(df)

    for col in ATT_COLS + WD_COLS + ["fa_marks", "sa_marks", "fa_rank", "sa_rank", "house_lat", "house_lon"]:
        df[col] = df[col].astype("float32")
    for col in ("n_rows_in_year", "n_schools_in_year"):
        df[col] = df[col].astype("int16")
    df["academic_year"] = df["academic_year"].astype("category")
    df["person_id"] = df["person_id"].astype("string[pyarrow]")
    df = df[CLEAN_COLUMNS].reset_index(drop=True)
    diag["rows_out"] = int(len(df))
    return df, diag


def _deduplicate(df: pd.DataFrame, diag: dict) -> pd.DataFrame:
    """One row per person_key. Main row = most attended days, then higher class, then higher school_key."""
    df = df.copy()
    df["_total_att"] = df[ATT_COLS].sum(axis=1, min_count=1).fillna(-1.0)
    df = df.sort_values(["person_key", "_total_att", "class_t", "school_key"],
                        ascending=[True, False, False, False], kind="mergesort")
    dup = df["person_key"].duplicated(keep=False)
    diag["persons_with_duplicates"] = int(df.loc[dup, "person_key"].nunique())
    main = df.drop_duplicates("person_key", keep="first").set_index("person_key")
    main["n_rows_in_year"] = 1
    main["n_schools_in_year"] = 1
    if dup.any():
        sub = df.loc[dup]
        monthly = sub.groupby("person_key")[ATT_COLS + WD_COLS].max()
        main.loc[monthly.index, ATT_COLS + WD_COLS] = monthly
        sizes = sub.groupby("person_key").agg(n_rows=("school_key", "size"), n_schools=("school_key", "nunique"))
        main.loc[sizes.index, "n_rows_in_year"] = sizes["n_rows"]
        main.loc[sizes.index, "n_schools_in_year"] = sizes["n_schools"]
        for att, wd in zip(ATT_COLS, WD_COLS):
            main[att] = main[att].where(~(main[att] > main[wd]), main[wd])
    return main.drop(columns=["_total_att"]).reset_index()
