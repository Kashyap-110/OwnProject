"""Extract the source table into the local Parquet cache (one file per academic year + a school file).

    python -m dropout_v2.extract --sample-pct 2     # quick dry run: the same 2% of students in every year
    python -m dropout_v2.extract                    # full extraction (about 9M rows per year)
    python -m dropout_v2.extract --years 2026-27 --refresh   # refresh the live year only
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path
from typing import Callable, Dict, List, Optional

from dropout_v2 import config, db, schema, utils
from dropout_v2.schema import RAW_SCHEMA, SCHOOL_SCHEMA

_TEXT = ["student_gender", "social_category", "rural_urban_flag", "ration_card_type", "dwelling_type", "is_deceased",
         "medium_name", "student_district_name", "student_mandal_name"]


def _select_list() -> List[str]:
    parts = {
        "person_id": "CAST(person_id AS STRING)",
        "student_id": "CAST(student_id AS STRING)",
        "academic_year": "CAST(academic_year AS STRING)",
        "class_currently_enrolled": "CAST(class_currently_enrolled AS INT)",
        "school_id": "CAST(school_id AS STRING)",
        "student_date_of_birth": "CAST(student_date_of_birth AS STRING)",
        "has_father_id": "CASE WHEN father_person_id IS NULL OR TRIM(CAST(father_person_id AS STRING)) = '' "
                         "THEN 0 ELSE 1 END",
        "has_mother_id": "CASE WHEN mother_person_id IS NULL OR TRIM(CAST(mother_person_id AS STRING)) = '' "
                         "THEN 0 ELSE 1 END",
        "house_latitude": "CAST(house_latitude AS DOUBLE)",
        "house_longitude": "CAST(house_longitude AS DOUBLE)",
        "src_is_dropout": "CAST(is_dropout AS STRING)",
        "fa_marks": "CAST(fa_marks AS DOUBLE)",
        "sa_marks": "CAST(sa_marks AS DOUBLE)",
    }
    for col in _TEXT:
        parts[col] = "CAST({} AS STRING)".format(col)
    for col in schema.MONTH_COLUMNS:
        parts[col] = "CAST({} AS DOUBLE)".format(col)
    missing = [c for c in schema.RAW_COLUMNS if c not in parts]
    if missing:
        raise RuntimeError("No SQL expression for raw columns {}".format(missing))
    return ["    {} AS {}".format(parts[c], c) for c in schema.RAW_COLUMNS]


def build_year_query(year: str, sample_pct: int) -> str:
    if year not in config.ACADEMIC_YEARS:
        raise ValueError("Unknown academic year {!r}".format(year))
    if not (0 <= int(sample_pct) <= 100):
        raise ValueError("sample_pct must be between 0 and 100")
    where = [
        "academic_year = '{}'".format(year),
        "class_currently_enrolled BETWEEN {} AND {}".format(config.KEEP_CLASS_MIN, config.KEEP_CLASS_MAX),
    ]
    if 0 < int(sample_pct) < 100:
        where.append("ABS(fnv_hash(CAST(person_id AS STRING))) % 100 < {}".format(int(sample_pct)))
    return "SELECT\n{}\nFROM {}\nWHERE {}".format(",\n".join(_select_list()), config.TABLE, "\n  AND ".join(where))


def build_school_query() -> str:
    parts = ["    CAST(school_id AS STRING) AS school_id"]
    for name, arrow_type in zip(SCHOOL_SCHEMA.names[1:], SCHOOL_SCHEMA.types[1:]):
        cast = "DOUBLE" if str(arrow_type) == "double" else "STRING"
        parts.append("    MAX(CAST({} AS {})) AS {}".format(name, cast, name))
    return "SELECT\n{}\nFROM {}\nWHERE school_id IS NOT NULL\nGROUP BY CAST(school_id AS STRING)".format(
        ",\n".join(parts), config.TABLE)


def run_extract(years: List[str], sample_pct: int, refresh: bool, batch_size: int,
                connect_fn: Callable = db.connect, root: Optional[Path] = None) -> Dict[str, object]:
    logger = utils.get_logger()
    tag = config.data_tag(sample_pct)
    raw_dir = utils.data_dir(tag, root) / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, object] = {}
    targets = [("schools", raw_dir / "schools.parquet", build_school_query(), SCHOOL_SCHEMA)]
    targets += [(y, raw_dir / "{}.parquet".format(y), build_year_query(y, sample_pct), RAW_SCHEMA) for y in years]
    for name, path, query, arrow_schema in targets:
        if path.exists() and not refresh:
            logger.info("%s: %s exists, skipped (use --refresh to extract again)", name, path)
            summary[name] = "skipped (exists)"
            continue
        started = time.time()
        logger.info("%s: extracting into %s", name, path)
        rows = db.fetch_to_parquet(query, path, arrow_schema, batch_size, connect_fn)
        logger.info("%s: %s rows in %.1f s", name, "{:,}".format(rows), time.time() - started)
        if rows == 0:
            path.unlink()
            raise RuntimeError("Query for {} returned no rows. Check the table and the filters:\n{}".format(name, query))
        summary[name] = rows
    utils.write_json(raw_dir / "extract_summary.json", {"tag": tag, "sample_pct": sample_pct, "results": summary})
    return summary


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Extract the source table into data/v2/<tag>/raw/.")
    parser.add_argument("--years", nargs="*", default=config.ACADEMIC_YEARS)
    parser.add_argument("--sample-pct", type=int, default=0, help="extract this percent of students (0 = all)")
    parser.add_argument("--refresh", action="store_true", help="extract again even if the file exists")
    parser.add_argument("--batch-size", type=int, default=200_000)
    args = parser.parse_args(argv)
    utils.setup_logging(None)
    try:
        summary = run_extract(args.years, args.sample_pct, args.refresh, args.batch_size)
    except Exception as exc:
        utils.get_logger().error("Extraction failed: %s\n%s", exc, traceback.format_exc())
        return 1
    print("\nExtraction finished: {}".format(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
