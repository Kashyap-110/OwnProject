"""Synthetic raw data with the real column layout and the real data defects.

Used by the unit tests and for a dry run of the whole pipeline without the database:
    python -m dropout_v2.train --synthetic
It is NOT a model of Andhra Pradesh; it only plants the patterns the pipeline must handle.
"""
from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from dropout_v2 import config, schema, utils

PLACEHOLDER_ID = "999999999999"
YEARS = config.ACADEMIC_YEARS
RAW_COLUMNS = schema.RAW_COLUMNS

_DISTRICTS = [
    ("EAST GODAVARI", ["RAJAHMUNDRY", "KORUKONDA"], (17.0, 81.8)),
    ("MANYAM", ["KURUPAM", "PALAKONDA"], (18.6, 83.5)),
    ("KURNOOL", ["ADONI", "YEMMIGANUR"], (15.6, 77.3)),
]
# management, category, lowest, highest, board, copies per district
_SCHOOL_TYPES = [
    ("MPP_ZPP SCHOOLS", "PRIMARY", -2, 5, "STATE", 2),
    ("MPP_ZPP SCHOOLS", "UPPER PR. AND SECONDARY", 6, 10, "STATE", 2),
    ("PVT.UNAIDED", "PR. UP PR. AND SECONDARY ONLY", -3, 10, "STATE", 2),
    ("PVT.UNAIDED (CBSE SYLLABUS)", "PR. UP PR. AND SECONDARY ONLY", -3, 10, "CBSE", 1),
    ("APSWREI SOCIETY SCHOOLS", "UPPER PR. AND SECONDARY", 6, 10, "STATE", 1),
    ("MUNCIPAL", "PRIMARY WITH UPPER PRIMARY", -2, 7, "STATE", 1),
    ("PVT.UNAIDED", "HIGHER SECONDARY ONLY", 11, 12, "STATE", 1),
]


def _make_schools(rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    k = 0
    for d_name, mandals, (lat, lon) in _DISTRICTS:
        for mgmt, cat, low, high, board, copies in _SCHOOL_TYPES:
            for _ in range(copies):
                k += 1
                rows.append({
                    "school_id": "28{:09d}".format(100000 + k * 37),
                    "school_name": "SCHOOL {}".format(k),
                    "school_management": mgmt,
                    "school_category": cat,
                    "board_affiliation": board,
                    "school_type_coed_boys_girls": "CO_ED",
                    "school_in_rural_urban": "Rural" if k % 3 else "Urban",
                    "total_students": float(rng.integers(80, 900)),
                    "total_teachers": 0.0 if mgmt.startswith("PVT") else float(rng.integers(4, 30)),
                    "lowest_class": float(low),
                    "highest_class": float(high),
                    "school_latitude": lat + rng.normal(0, 0.05),
                    "school_longitude": lon + rng.normal(0, 0.05),
                    "school_district_name": d_name,
                    "school_mandal_name": mandals[k % 2],
                })
    return pd.DataFrame(rows, columns=schema.SCHOOL_COLUMNS)


def _hazard(propensity: float, over_age: int, cls: int) -> float:
    logit = -4.6 + 7.0 * (0.85 - propensity) + 0.7 * over_age + 0.4 * (cls >= 7)
    return 1.0 / (1.0 + np.exp(-logit))


def _pick_school(rng, schools: pd.DataFrame, district: str, cls: int, current: Optional[str]) -> str:
    if current is not None:
        row = schools.loc[schools["school_id"] == current].iloc[0]
        if row["lowest_class"] <= cls <= row["highest_class"]:
            return current
    ok = schools[(schools["school_district_name"] == district)
                 & (schools["lowest_class"] <= cls) & (schools["highest_class"] >= cls)]
    return str(ok["school_id"].iloc[int(rng.integers(len(ok)))])


def _month_values(rng, year: str, rate: float, drop_month: Optional[int], removed: bool,
                  school_silent: bool) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {}
    for idx, month in enumerate(config.MONTHS):
        wd: Optional[float]
        att: Optional[float]
        if month == "may":
            wd, att = 0.0, 0.0
        else:
            wd = float(rng.integers(18, 27))
            month_rate = min(1.0, max(0.0, rate + rng.normal(0, 0.05)))
            if drop_month is not None:
                if idx >= drop_month:
                    month_rate = 0.0
                elif idx >= drop_month - 2:
                    month_rate *= 0.5
            att = float(rng.binomial(int(wd), month_rate))
            if drop_month is not None and idx >= drop_month and removed:
                wd, att = None, None
        if year == "2022-23" and month in ("jun", "jul"):
            wd, att = None, None
        if year == config.LIVE_YEAR and idx >= 2:
            wd, att = 0.0, 0.0            # the real table stores future months as 0/0
        if school_silent:
            wd, att = None, None
        out["{}_attended".format(month)] = att
        out["{}_working_days".format(month)] = wd
    return out


def _empty_months() -> Dict[str, Optional[float]]:
    return {c: None for c in schema.MONTH_COLUMNS}


def _marks(rng, year: str, cls: int, rate: float, weak: bool) -> Tuple[Optional[float], Optional[float]]:
    if year == config.LIVE_YEAR:
        return None, None
    base_fa = 300 + 60 * max(cls, 1)
    base_sa = 200 + 50 * max(cls, 1)
    values = []
    for base in (base_fa, base_sa):
        v = base * rate * rng.uniform(0.75, 1.05) * (0.5 if weak else 1.0)
        u = rng.random()
        if u < 0.02:
            v = 0.0
        elif u < 0.10:
            v = None
        elif u < 0.103:
            v = v * 80.0          # impossible outlier, as seen in 2025-26
        values.append(v)
    return values[0], values[1]


def _choice(rng, options: List, probs: List[float]):
    return options[int(rng.choice(len(options), p=probs))]


def _simulate_student(i: int, rng, schools: pd.DataFrame, rows: Dict[str, List[dict]],
                      silent_school: str) -> None:
    base = (10 ** 10 if i % 2 == 0 else 10 ** 11) + i * 7919003
    pid = "{:012d}".format(base)
    new_pid = "{:012d}".format(base + 555)
    sid = str(3000000000 + i)
    d_name, mandals, (lat, lon) = _DISTRICTS[int(rng.integers(len(_DISTRICTS)))]
    mandal = mandals[int(rng.integers(2))]
    entry = 0 if rng.random() < 0.85 else int(rng.integers(1, 5))
    if entry == 0:
        cls = int(rng.integers(-2, 10))
        if cls == 0:
            cls = 1
    else:
        cls = 1 if rng.random() < 0.7 else int(rng.integers(2, 9))
    over_age = int(_choice(rng, [0, 1, 2, 3], [0.8, 0.12, 0.06, 0.02]))
    propensity = float(np.clip(rng.beta(9, 1.6), 0.2, 0.99))
    gender = _choice(rng, ["Male", "Female", "Transgender", "Others", None], [0.49, 0.49, 0.005, 0.005, 0.01])
    social = _choice(rng, ["BC", "SC", "ST", "OC", "null", None], [0.45, 0.18, 0.08, 0.14, 0.05, 0.10])
    ration = _choice(rng, ["BPL", "AAY", "APL", "null", None], [0.80, 0.05, 0.05, 0.05, 0.05])
    dwelling = _choice(rng, ["Puccahouse-Daba", "Pakkahouse", "RCChouse", "Kacchahouse", "Foilhouse", "null", None],
                       [0.3, 0.15, 0.15, 0.1, 0.1, 0.1, 0.1])
    rural = _choice(rng, ["R", "U", None], [0.7, 0.28, 0.02])
    deceased = _choice(rng, [None, "Deceased", "reported deceased"], [0.992, 0.005, 0.003])
    has_father = int(rng.random() < 0.92)
    has_mother = int(rng.random() < 0.95)
    house_lat = lat + rng.normal(0, 0.06) if rng.random() > 0.2 else None
    house_lon = lon + rng.normal(0, 0.06) if house_lat is not None else None
    entry_start = config.year_start(YEARS[entry])
    age = max(3, cls + 5 + over_age)
    dob_date = _dt.date(entry_start - age, 6, 1) - _dt.timedelta(days=int(rng.integers(0, 365)))
    u = rng.random()
    dob = None if u < 0.02 else ("1963-01-01" if u < 0.023 else dob_date.isoformat())
    change_pid_from = 3 if rng.random() < 0.01 else None   # person_id changes in 2025-26, student_id stays

    status = "active"
    ghost_left = 0
    school = _pick_school(rng, schools, d_name, cls, None)
    for t in range(entry, len(YEARS)):
        year = YEARS[t]
        this_pid = new_pid if (change_pid_from is not None and t >= change_pid_from) else pid
        medium = "English" if rng.random() < 0.85 else "Telugu"
        common = {
            "person_id": this_pid, "student_id": sid, "academic_year": year,
            "student_date_of_birth": dob, "student_gender": gender, "social_category": social,
            "rural_urban_flag": rural, "ration_card_type": ration, "dwelling_type": dwelling,
            "has_father_id": has_father, "has_mother_id": has_mother, "is_deceased": deceased,
            "medium_name": medium, "student_district_name": d_name, "student_mandal_name": mandal,
            "house_latitude": house_lat, "house_longitude": house_lon,
        }
        if status == "active":
            drop_now = rng.random() < _hazard(propensity, over_age, cls) and cls <= 10
            drop_month = int(rng.integers(1, 10)) if drop_now else None
            removed = bool(rng.random() < 0.5)
            silent = school == silent_school and year in ("2025-26", "2026-27")
            junior_college = cls >= 11
            if junior_college:
                months = _empty_months()     # junior colleges do not report attendance here
                fa, sa = None, None
            else:
                months = _month_values(rng, year, propensity, drop_month, removed, silent)
                fa, sa = _marks(rng, year, cls, propensity, weak=drop_now and (drop_month or 99) < 8)
            row = dict(common, class_currently_enrolled=cls, school_id=school,
                       src_is_dropout="true" if (drop_now and rng.random() < 0.4) else "false",
                       fa_marks=fa, sa_marks=sa, **months)
            if (not junior_college) and year != config.LIVE_YEAR and rng.random() < 0.01:
                other = _pick_school(rng, schools[schools["school_id"] != school], d_name, cls, None)
                second = dict(row, school_id=other)
                for idx, month in enumerate(config.MONTHS):
                    if idx < 5:
                        second["{}_attended".format(month)] = None
                        second["{}_working_days".format(month)] = None
                    else:
                        row["{}_attended".format(month)] = None
                        row["{}_working_days".format(month)] = None
                rows[year].append(second)
            rows[year].append(row)
            if deceased == "Deceased" and rng.random() < 0.4:
                status = "dead"
            elif drop_now:
                status = "gone"
                ghost_left = int(rng.integers(1, 4)) if rng.random() < 0.7 else 0
            else:
                if cls >= 10:
                    cls += 1
                    if cls > 12:
                        status = "done"
                elif cls == -1:
                    cls = 1
                elif rng.random() < 0.96:
                    cls += 1
                if status == "active":
                    school = _pick_school(rng, schools, d_name, cls, school)
        elif status == "gone" and ghost_left > 0:
            ghost_left -= 1
            rows[year].append(dict(common, class_currently_enrolled=cls, school_id=school,
                                   src_is_dropout="true" if rng.random() < 0.5 else "false",
                                   fa_marks=None, sa_marks=None, **_empty_months()))


def _to_frame(rows: List[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    for col in RAW_COLUMNS:
        if col not in df.columns:
            df[col] = None
    df = df[RAW_COLUMNS].copy()
    df["class_currently_enrolled"] = df["class_currently_enrolled"].astype("Int32")
    for col in ["has_father_id", "has_mother_id"]:
        df[col] = df[col].astype("Int32")
    for col in ["house_latitude", "house_longitude", "fa_marks", "sa_marks"] + schema.MONTH_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    return df


def make_raw_years(n_students: int = 3000, seed: int = 0) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """Return ({academic_year: raw frame}, schools frame) with planted data defects."""
    rng = np.random.default_rng(seed)
    schools = _make_schools(rng)
    silent_school = str(schools.loc[schools["school_category"] == "PRIMARY", "school_id"].iloc[0])
    rows: Dict[str, List[dict]] = {y: [] for y in YEARS}
    for i in range(n_students):
        _simulate_student(i, rng, schools, rows, silent_school)
    frames = {y: _to_frame(rows[y]) for y in YEARS}

    # Placeholder person_id: 40 rows of 2024-25 share one ID (the real table has 127,571).
    f = frames["2024-25"]
    candidates = f.index[f["class_currently_enrolled"].between(1, 9)]
    chosen = rng.choice(candidates, size=min(40, len(candidates)), replace=False)
    f.loc[chosen, "person_id"] = PLACEHOLDER_ID

    # Known gap: most 2023-24 class 1 and pre-primary rows are missing.
    g = frames["2023-24"]
    in_gap = g["class_currently_enrolled"].isin(config.KNOWN_GAPS["2023-24"])
    drop = in_gap & (rng.random(len(g)) < 0.9)
    frames["2023-24"] = g.loc[~drop].reset_index(drop=True)
    return frames, schools


def write_synthetic_cache(tag: str = "synthetic", n_students: int = 3000, seed: int = 0,
                          root: Optional[Path] = None) -> Path:
    """Write synthetic raw files in the same layout as `extract` writes real data."""
    out = utils.data_dir(tag, root)
    raw_dir = out / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    frames, schools = make_raw_years(n_students=n_students, seed=seed)
    for year, frame in frames.items():
        table = pa.Table.from_pandas(frame, schema=schema.RAW_SCHEMA, preserve_index=False)
        pq.write_table(table, str(raw_dir / "{}.parquet".format(year)))
    school_table = pa.Table.from_pandas(schools, schema=schema.SCHOOL_SCHEMA, preserve_index=False)
    pq.write_table(school_table, str(raw_dir / "schools.parquet"))
    return out
