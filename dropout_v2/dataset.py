"""DataStore: file layout and caching of cleaned frames, labels and snapshot datasets.

data/v2/<tag>/raw/<year>.parquet, raw/schools.parquet          <- written by extract.py
data/v2/<tag>/derived/<sig>/clean/<year>.parquet (+ .json)     <- clean.clean_year
data/v2/<tag>/derived/<sig>/labels/<year>.parquet              <- labels.build_labels
data/v2/<tag>/derived/<sig>/datasets/<snapshot>/<year>.parquet <- features.build_features + label

<sig> is a short hash of the rules that change derived data, so a rule change never reuses stale files.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import dropout_v2
from dropout_v2 import config, features, schema, utils
from dropout_v2.clean import clean_year
from dropout_v2.labels import build_labels


def rules_signature() -> str:
    rules = {
        "version": dropout_v2.__version__,
        "label_window": config.LABEL_WINDOW,
        "school_reporting_min": config.SCHOOL_REPORTING_MIN,
        "class_continued_min": config.CLASS_CONTINUED_MIN,
        "base_classes": list(config.BASE_CLASSES),
        "keep_class": [config.KEEP_CLASS_MIN, config.KEEP_CLASS_MAX],
        "max_rows_per_person_year": config.MAX_ROWS_PER_PERSON_YEAR,
        "deceased": sorted(config.DECEASED_VALUES),
        "known_gaps": config.KNOWN_GAPS,
        "use_social_category": config.USE_SOCIAL_CATEGORY,
        "smoothing": config.RATE_SMOOTHING_WEIGHT,
        "marks_outlier": config.MARKS_OUTLIER_FACTOR,
    }
    return hashlib.sha1(json.dumps(rules, sort_keys=True).encode("utf-8")).hexdigest()[:10]


def _string_mapper(arrow_type):
    if pa.types.is_string(arrow_type) or pa.types.is_large_string(arrow_type):
        return pd.StringDtype("pyarrow")
    return None


class DataStore:
    def __init__(self, tag: str, root: Optional[Path] = None):
        self.tag = tag
        self.base_dir = utils.data_dir(tag, root)
        self._clean: Dict[str, pd.DataFrame] = {}
        self._labels: Dict[str, Optional[pd.DataFrame]] = {}
        self._schools: Optional[pd.DataFrame] = None
        self.diagnostics: Dict[str, dict] = {}

    # ------------------------------------------------------------------ paths
    @property
    def derived_dir(self) -> Path:
        return self.base_dir / "derived" / rules_signature()

    def raw_path(self, year: str) -> Path:
        return self.base_dir / "raw" / "{}.parquet".format(year)

    def schools_path(self) -> Path:
        return self.base_dir / "raw" / "schools.parquet"

    def _clean_path(self, year: str) -> Path:
        return self.derived_dir / "clean" / "{}.parquet".format(year)

    def _labels_path(self, year: str) -> Path:
        return self.derived_dir / "labels" / "{}.parquet".format(year)

    def _dataset_path(self, year: str, snapshot: str) -> Path:
        return self.derived_dir / "datasets" / snapshot / "{}.parquet".format(year)

    def years_available(self) -> List[str]:
        return [y for y in config.ACADEMIC_YEARS if self.raw_path(y).exists() or self._clean_path(y).exists()]

    @staticmethod
    def _fresh(path: Path, sources: List[Path]) -> bool:
        """A cached file is usable only when it is newer than every raw file it was built from."""
        if not path.exists():
            return False
        built = path.stat().st_mtime
        return all(built >= src.stat().st_mtime for src in sources if src.exists())

    def _raw_sources(self, years: List[Optional[str]]) -> List[Path]:
        return [self.raw_path(y) for y in years if y is not None]

    # ------------------------------------------------------------------ raw and clean
    def read_raw(self, year: str) -> pd.DataFrame:
        path = self.raw_path(year)
        if not path.exists():
            raise FileNotFoundError(
                "Raw data for {} not found at {}. Run: python -m dropout_v2.extract{}".format(
                    year, path, "" if self.tag == "full" else "  (with the same --sample-pct / --tag)"))
        names = pq.read_schema(str(path)).names
        dictionary = [c for c in schema.RAW_DICTIONARY_COLUMNS if c in names]
        table = pq.read_table(str(path), read_dictionary=dictionary)
        missing = [c for c in schema.RAW_COLUMNS if c not in table.schema.names]
        if missing:
            raise ValueError("Raw file {} is missing columns {}. Re-run extract with --refresh.".format(path, missing))
        return table.to_pandas(types_mapper=_string_mapper)

    def clean(self, year: str) -> Optional[pd.DataFrame]:
        """Cleaned frame for a year, or None when the year is not part of the data."""
        if year is None:
            return None
        if year in self._clean:
            return self._clean[year]
        path = self._clean_path(year)
        diag_path = path.with_suffix(".json")
        if self._fresh(path, self._raw_sources([year])):
            frame = pd.read_parquet(str(path))
            if diag_path.exists():
                self.diagnostics[year] = json.loads(diag_path.read_text(encoding="utf-8"))
        else:
            if not self.raw_path(year).exists() and year not in config.ACADEMIC_YEARS:
                return None
            with utils.step("clean {}".format(year)):
                frame, diag = clean_year(self.read_raw(year), year)
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(str(path), index=False)
            utils.write_json(diag_path, diag)
            self.diagnostics[year] = diag
        self._clean[year] = frame
        return frame

    def frames(self, years: List[Optional[str]]) -> Dict[str, pd.DataFrame]:
        """Cleaned frames for the given years that exist in the data."""
        out = {}
        for year in years:
            if year is not None and year in self.years_available():
                out[year] = self.clean(year)
        return out

    def schools(self) -> pd.DataFrame:
        if self._schools is None:
            path = self.schools_path()
            if not path.exists():
                raise FileNotFoundError("School file not found at {}. Run: python -m dropout_v2.extract".format(path))
            raw = pq.read_table(str(path)).to_pandas(types_mapper=_string_mapper)
            self._schools = features.prepare_school_dim(raw)
        return self._schools

    # ------------------------------------------------------------------ labels and datasets
    def labels(self, year: str) -> Optional[pd.DataFrame]:
        """Labels of base year `year` (None when year T+1 is not in the data)."""
        if year in self._labels:
            return self._labels[year]
        nxt = config.next_year(year)
        available = self.years_available()
        if year not in available or nxt is None or nxt not in available:
            self._labels[year] = None
            return None
        path = self._labels_path(year)
        if self._fresh(path, self._raw_sources([year, nxt])):
            lab = pd.read_parquet(str(path))
        else:
            with utils.step("labels {}".format(year)):
                base = features.base_population(self.clean(year))
                lab = build_labels(base, self.clean(nxt))
                lab["school_key"] = base["school_key"].to_numpy()
                lab["class_t"] = base["class_t"].to_numpy()
                lab["district"] = base["district"].to_numpy()
                lab["mandal_key"] = (base["district"].astype(str) + "|" + base["mandal"].astype(str)).to_numpy()
                lab["src_is_dropout"] = base["src_is_dropout"].to_numpy()
                lab = lab.reset_index(drop=True)
            path.parent.mkdir(parents=True, exist_ok=True)
            lab.to_parquet(str(path), index=False)
        self._labels[year] = lab
        return lab

    def dataset(self, year: str, snapshot: str, with_label: bool = True) -> pd.DataFrame:
        """Features (+ label, label_reason) of base year `year` at `snapshot`."""
        path = self._dataset_path(year, snapshot)
        sources = self._raw_sources([config.prev_year(year, 2), config.prev_year(year), year,
                                     config.next_year(year)]) + [self.schools_path()]
        if self._fresh(path, sources):
            ds = pd.read_parquet(str(path))
        else:
            with utils.step("features {} {}".format(year, snapshot)):
                frames = self.frames([config.prev_year(year, 2), config.prev_year(year), year])
                prev = config.prev_year(year)
                prev_labels = self.labels(prev) if (prev is not None and config.visible_months(snapshot)) else None
                ds = features.build_features(year, snapshot, frames, self.schools(), prev_labels)
                lab = self.labels(year)
                if lab is not None:
                    ds = ds.merge(lab[["person_key", "label", "label_reason"]], on="person_key", how="left",
                                  validate="one_to_one")
            path.parent.mkdir(parents=True, exist_ok=True)
            ds.to_parquet(str(path), index=False)
        if not with_label:
            ds = ds.drop(columns=[c for c in ("label", "label_reason") if c in ds.columns])
        return ds

    # ------------------------------------------------------------------ reporting
    def profile(self) -> pd.DataFrame:
        """Per-year data-quality table from cleaning diagnostics and cleaned frames."""
        rows = []
        for year in self.years_available():
            frame = self.clean(year)
            diag = self.diagnostics.get(year, {})
            base = features.base_population(frame)
            row = {
                "academic_year": year,
                "rows_in": diag.get("rows_in"),
                "rows_bad_person_id": diag.get("rows_bad_person_id"),
                "rows_bad_class": diag.get("rows_bad_class"),
                "rows_placeholder": diag.get("rows_placeholder"),
                "persons_with_duplicates": diag.get("persons_with_duplicates"),
                "students_clean": int(len(frame)),
                "base_students_class_1_9": int(len(base)),
                "deceased_excluded": int(frame["deceased"].sum()),
            }
            for month in config.MONTHS[:11]:
                row["wd_{}".format(month)] = float(base["wd_{}".format(month)].notna().mean()) if len(base) else None
            rows.append(row)
        return pd.DataFrame(rows)
