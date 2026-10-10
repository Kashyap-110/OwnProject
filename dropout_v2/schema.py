"""Column layout of the raw extraction files (single source for extract.py and synthetic.py)."""
from __future__ import annotations

import pyarrow as pa

from dropout_v2.config import MONTHS

MONTH_COLUMNS = []
for _m in MONTHS:
    MONTH_COLUMNS += ["{}_attended".format(_m), "{}_working_days".format(_m)]

_RAW_FIELDS = [
    ("person_id", pa.string()),
    ("student_id", pa.string()),
    ("academic_year", pa.string()),
    ("class_currently_enrolled", pa.int32()),
    ("school_id", pa.string()),
    ("student_date_of_birth", pa.string()),
    ("student_gender", pa.string()),
    ("social_category", pa.string()),
    ("rural_urban_flag", pa.string()),
    ("ration_card_type", pa.string()),
    ("dwelling_type", pa.string()),
    ("has_father_id", pa.int32()),
    ("has_mother_id", pa.int32()),
    ("is_deceased", pa.string()),
    ("medium_name", pa.string()),
    ("student_district_name", pa.string()),
    ("student_mandal_name", pa.string()),
    ("house_latitude", pa.float64()),
    ("house_longitude", pa.float64()),
    ("src_is_dropout", pa.string()),
] + [(c, pa.float64()) for c in MONTH_COLUMNS] + [
    ("fa_marks", pa.float64()),
    ("sa_marks", pa.float64()),
]

RAW_SCHEMA = pa.schema(_RAW_FIELDS)
RAW_COLUMNS = [name for name, _ in _RAW_FIELDS]

# Text columns with few distinct values: read as dictionary (pandas category) to save memory.
RAW_DICTIONARY_COLUMNS = [
    "academic_year", "student_gender", "social_category", "rural_urban_flag", "ration_card_type",
    "dwelling_type", "is_deceased", "medium_name", "student_district_name", "student_mandal_name",
    "src_is_dropout",
]

_SCHOOL_FIELDS = [
    ("school_id", pa.string()),
    ("school_name", pa.string()),
    ("school_management", pa.string()),
    ("school_category", pa.string()),
    ("board_affiliation", pa.string()),
    ("school_type_coed_boys_girls", pa.string()),
    ("school_in_rural_urban", pa.string()),
    ("total_students", pa.float64()),
    ("total_teachers", pa.float64()),
    ("lowest_class", pa.float64()),
    ("highest_class", pa.float64()),
    ("school_latitude", pa.float64()),
    ("school_longitude", pa.float64()),
    ("school_district_name", pa.string()),
    ("school_mandal_name", pa.string()),
]

SCHOOL_SCHEMA = pa.schema(_SCHOOL_FIELDS)
SCHOOL_COLUMNS = [name for name, _ in _SCHOOL_FIELDS]
