"""Builders for small hand-made raw and cleaned frames used across tests."""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from dropout_v2 import config, schema


def raw_row(**overrides) -> dict:
    """One raw row with sensible defaults; full attendance Jun-Apr (22 working days, 20 attended)."""
    row = {
        "person_id": "000000000001",
        "student_id": "3000000001",
        "academic_year": "2024-25",
        "class_currently_enrolled": 3,
        "school_id": "28000000001",
        "student_date_of_birth": "2016-03-01",
        "student_gender": "Male",
        "social_category": "BC",
        "rural_urban_flag": "R",
        "ration_card_type": "BPL",
        "dwelling_type": "Pakkahouse",
        "has_father_id": 1,
        "has_mother_id": 1,
        "is_deceased": None,
        "medium_name": "English",
        "student_district_name": "EAST GODAVARI",
        "student_mandal_name": "KORUKONDA",
        "house_latitude": 17.1,
        "house_longitude": 81.8,
        "src_is_dropout": "false",
        "fa_marks": 500.0,
        "sa_marks": 300.0,
    }
    for month in config.MONTHS:
        wd = 0.0 if month == "may" else 22.0
        att = 0.0 if month == "may" else 20.0
        row["{}_attended".format(month)] = att
        row["{}_working_days".format(month)] = wd
    row.update(overrides)
    return row


def raw_frame(rows: List[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df = df[schema.RAW_COLUMNS].copy()
    df["class_currently_enrolled"] = df["class_currently_enrolled"].astype("Int32")
    for col in ["house_latitude", "house_longitude", "fa_marks", "sa_marks"] + schema.MONTH_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    return df


def set_months(row: dict, attended: Dict[str, Optional[float]], working: Optional[Dict[str, Optional[float]]] = None) -> dict:
    """Overwrite monthly values; months not given keep their value."""
    for month, value in attended.items():
        row["{}_attended".format(month)] = value
    for month, value in (working or {}).items():
        row["{}_working_days".format(month)] = value
    return row


def no_attendance(row: dict) -> dict:
    for month in config.MONTHS:
        row["{}_attended".format(month)] = None
        row["{}_working_days".format(month)] = None
    return row


def school_frame(rows: Optional[List[dict]] = None) -> pd.DataFrame:
    default = {
        "school_id": "28000000001", "school_name": "SCHOOL 1", "school_management": "MPP_ZPP SCHOOLS",
        "school_category": "PRIMARY", "board_affiliation": "STATE", "school_type_coed_boys_girls": "CO_ED",
        "school_in_rural_urban": "Rural", "total_students": 300.0, "total_teachers": 10.0,
        "lowest_class": 1.0, "highest_class": 5.0, "school_latitude": 17.1, "school_longitude": 81.81,
        "school_district_name": "EAST GODAVARI", "school_mandal_name": "KORUKONDA",
    }
    rows = rows or [default]
    filled = [dict(default, **r) for r in rows]
    return pd.DataFrame(filled, columns=schema.SCHOOL_COLUMNS)
