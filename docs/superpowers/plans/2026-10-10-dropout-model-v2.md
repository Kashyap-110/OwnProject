# Dropout Early-Warning Model v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Execution note (2026-10-10):** The user asked to "plan and implement", with a demo on Monday. This plan is executed inline in the same session (executing-plans). Commit steps are skipped: the repository has no commits yet, and commits happen only when the user asks.

**Goal:** A runnable package `dropout_v2/` that extracts the new table, refines it in code, trains one calibrated LightGBM model per quarterly snapshot (classes 1–9), evaluates it on a later year, and scores the live year with tiers and reasons.

**Architecture:** Pure-function modules (clean → labels → features) operate on one academic year at a time and are cached as Parquet. CLIs (`extract`, `train`, `score`, `selftest`) orchestrate them. A synthetic data generator drives all unit and end-to-end tests, so the pipeline is verified without the database.

**Tech Stack:** Python 3.9, pandas 2.3.3, numpy 2.0.2, pyarrow 21.0.0, scikit-learn 1.6.1, lightgbm 4.6.0, matplotlib 3.9.4, joblib 1.5.3, impyla 0.24.0, unittest.

**Spec:** `docs/superpowers/specs/2026-10-10-dropout-model-v2-design.md`

## Global Constraints

- Python 3.9 syntax only: no `match`, no `X | Y` unions at runtime, no `zip(strict=)`. Every module starts with `from __future__ import annotations`.
- Only packages already on the server (`requirements.txt`). No pytest; tests use `unittest`.
- Model scope: base classes 1–9. Class 10 is excluded. Classes −3..12 are kept in the cleaned data for history and labels.
- Source table: `aiml_usecase_curated.final_student_dropout_info_new`. Years: 2022-23, 2023-24, 2024-25, 2025-26, 2026-27 (live).
- Split: train 2023-24, validation 2024-25, test 2025-26, live 2026-27. 2022-23 is history only (flag to include).
- Label window `["jun", "jul"]`. A school reports attendance when its share of rows with working days in the window is ≥ 0.5.
- Snapshots: `start`=0, `jul`=2, `aug`=3, `nov`=6, `feb`=9, `apr`=11 visible months from June. Current-year marks and `n_rows_in_year` only in `apr`.
- Never use as features: `highest_qualification`, `student_age`, `is_dropout`, `dropout_reason`, `marital_status`, `members_count`, `siblings_below_5`, `income_band`, health columns, `child_status`, `thalliki_vandanam`, kit columns, `religion`, `caste_name`.
- IDs: `person_key` int64 (digits of `person_id`) for joins; the original `person_id` string is kept for output.
- Errors raise with clear messages; no silent empty frames.
- Data under `data/v2/<tag>/`; artefacts under `artifacts/v2/<run_id>/`. The tag is `full`, `sample<N>`, or `synthetic`.

## Review Focus

1. **impyla returns unexpected Python types** (`Decimal`, `datetime.date`, `bytes`, numeric strings). The batch converter must coerce them, not crash. Pinned in Task 11 (`test_rows_to_table_coerces_types`).
2. **Future months stored as 0/0 instead of NULL** in the live year. Snapshot auto-selection and attendance features must treat 0 working days as "no data". Pinned in Task 4 (`test_zero_working_days_is_no_data`) and Task 12 (`test_detect_available_months_ignores_zero_months`).
3. **Unseen category at scoring** (new district or management value). It must map to `OTHER`, not crash or shift codes. Pinned in Task 8 (`test_unseen_category_maps_to_other`).
4. **Degenerate evaluation subsets** (a group with one class or a few rows). Metrics return NaN, not an exception. Pinned in Task 9 (`test_breakdown_handles_single_class_group`).
5. **Memory on full data** (about 9M rows per year). The cleaned frame must use compact dtypes (float32, int8/int16, category). Pinned in Task 3 (`test_clean_output_dtypes_are_compact`).

---

## File Structure

| File | Responsibility |
|---|---|
| `dropout_v2/__init__.py` | Package marker, version string |
| `dropout_v2/config.py` | All constants (years, months, snapshots, label rules, maps, model params, paths, Impala settings) |
| `dropout_v2/reference.py` | Static reference data: tribal mandals, school-management group map |
| `dropout_v2/utils.py` | Logging setup, step timer, memory log, Markdown table, JSON writer, run-dir helpers |
| `dropout_v2/synthetic.py` | Synthetic raw yearly frames with planted patterns; writer for a synthetic raw cache |
| `dropout_v2/clean.py` | Raw year → cleaned one-row-per-student frame + diagnostics |
| `dropout_v2/attendance.py` | Monthly arrays + visible months → attendance feature arrays |
| `dropout_v2/labels.py` | Base year + next year → label, label_reason; school reporting share |
| `dropout_v2/features.py` | Base year + snapshot → feature frame; feature column lists |
| `dropout_v2/dataset.py` | `DataStore`: paths, cached clean frames, labels, and datasets |
| `dropout_v2/modeling.py` | Matrix prep with fixed categories, negative sampling, LightGBM fit, calibration, bundle save/load |
| `dropout_v2/evaluate.py` | Metrics, top-k, calibration table, breakdowns, bootstrap, baselines, single-feature AUC, plots |
| `dropout_v2/explain.py` | `pred_contrib` → top-3 reasons; global importance |
| `dropout_v2/report.py` | MODEL_REPORT.md and SCORING_SUMMARY.md writers |
| `dropout_v2/train.py` | Training CLI |
| `dropout_v2/db.py` | Impala connection; streaming fetch into Parquet; batch → Arrow conversion |
| `dropout_v2/extract.py` | Extraction CLI and SQL builders |
| `dropout_v2/score.py` | Scoring CLI, month detection, tiers, outputs, legacy export |
| `dropout_v2/selftest.py` | Runs the unittest suite |
| `dropout_v2/tests/test_*.py` | One test module per source module + end-to-end |
| `docs/dropout_v2_RUNBOOK.md` | Server run guide for the user |

---

### Task 1: Local test environment, package skeleton, config, utils

**Files:**
- Create: `dropout_v2/__init__.py`, `dropout_v2/config.py`, `dropout_v2/reference.py`, `dropout_v2/utils.py`, `dropout_v2/selftest.py`, `dropout_v2/tests/__init__.py`, `dropout_v2/tests/test_config.py`

**Interfaces:**
- Produces:
  - `config.MONTHS: list[str]` (jun..may), `config.SNAPSHOTS: dict[str, int]`, `config.visible_months(snapshot) -> list[str]`, `config.year_start(year) -> int`, `config.prev_year(year, n=1) -> str | None`, `config.next_year(year) -> str | None`, `config.BASE_CLASSES: tuple[int, ...]`
  - `utils.setup_logging(run_dir: Path | None) -> logging.Logger`, `utils.step(name)` (context manager that logs the duration), `utils.md_table(df: pd.DataFrame, floatfmt: str = ".4f") -> str`, `utils.write_json(path, obj)`, `utils.data_dir(tag) -> Path`, `utils.new_run_dir(tag) -> Path`

- [ ] Create a `uv` venv in the scratchpad with Python 3.9 and the server's exact versions.
- [ ] Write `test_config.py`:
  - `visible_months("jul") == ["jun", "jul"]`; `visible_months("start") == []`
  - `prev_year("2024-25") == "2023-24"`; `prev_year("2022-23") is None`; `next_year("2026-27") is None`
  - `BASE_CLASSES == (1, ..., 9)`
- [ ] Implement `config.py`, `reference.py`, `utils.py`, `selftest.py`. Run the tests; they pass.

### Task 2: Synthetic data generator

**Files:**
- Create: `dropout_v2/synthetic.py`, `dropout_v2/tests/test_synthetic.py`

**Interfaces:**
- Produces:
  - `synthetic.RAW_COLUMNS: list[str]` (identical to the extraction output columns)
  - `synthetic.make_raw_years(n_students=3000, seed=0) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]` (years → raw frames, schools frame)
  - `synthetic.write_synthetic_cache(tag="synthetic", n_students=3000, seed=0) -> Path`
- Planted cases:
  - leading-zero `person_id`s;
  - a placeholder `person_id` "999999999999" with 40 rows in 2024-25;
  - duplicate rows (transfers);
  - deceased students;
  - one school with no attendance in 2025-26;
  - class 10 → 11 rows without attendance;
  - `person_id` change with the same `student_id`;
  - carried-forward rows with no attendance after a dropout;
  - NULL tokens;
  - marks outliers;
  - 2023-24 class 1 / pre-primary mostly missing;
  - live year 2026-27 with Jun–Jul attendance only (later months 0/0, marks NULL).
- Signal: dropout probability rises with over-age and low attendance propensity; attendance falls before the dropout.

- [ ] Write tests:
  - all years present;
  - columns == `RAW_COLUMNS`;
  - the placeholder has 40 rows in 2024-25;
  - 2026-27 Aug–May working days are all 0;
  - leading zeros are present.
- [ ] Implement. Run the tests; they pass.

### Task 3: Cleaning

**Files:**
- Create: `dropout_v2/clean.py`, `dropout_v2/tests/test_clean.py`

**Interfaces:**
- Consumes: `config.*`, raw frame with `RAW_COLUMNS`
- Produces:
  - `clean.clean_year(raw: pd.DataFrame, year: str) -> tuple[pd.DataFrame, dict]`
  - Output columns:
    - IDs and year: `person_key` int64, `person_id` string, `student_key` int64 (−1 = missing), `school_key` int64 (−1 = missing), `academic_year`, `class_t` int8
    - profile: `dob` datetime64, `gender`, `social_category`, `rural_urban`, `ration_card`, `dwelling`, `medium`, `district`, `mandal` (category), `has_father_id`, `has_mother_id` int8, `deceased` bool
    - location: `house_lat`, `house_lon` float32
    - comparison only: `src_is_dropout` category
    - attendance: `att_<m>`, `wd_<m>` float32 for 12 months (NaN = no data)
    - marks: `fa_marks`, `sa_marks`, `fa_rank`, `sa_rank` float32; `fa_zero`, `sa_zero`, `marks_missing` int8
    - duplicates: `n_rows_in_year`, `n_schools_in_year` int16
  - Diagnostics keys: `rows_in`, `rows_bad_person_id`, `rows_bad_class`, `placeholder_ids` (dict id → rows), `rows_placeholder`, `persons_with_duplicates`, `rows_out`
  - `clean.to_key(values: pd.Series) -> pd.Series[int64]`, `clean.clean_text(values: pd.Series, mapping: dict | None = None) -> pd.Series[category]`

- [ ] Write tests:
  - `to_key` keeps digits and maps junk to −1; `"096904697373" -> 96904697373`
  - the placeholder rows are removed and recorded
  - duplicates collapse to one row: the main row has the most attended days, monthly values are the max across rows, `n_rows_in_year` = 2
  - class 0 and 13 are dropped
  - `wd == 0` → NaN, attended is clipped to working days, working days > 31 → NaN
  - NULL tokens → "UNKNOWN"; gender "Transgender"/"Others" → "OTHER"
  - marks outlier (> 3 × p99 within the class) → NaN; ranks are in (0, 1]
  - `test_clean_output_dtypes_are_compact`: att/wd are float32, `class_t` int8, text columns are category
- [ ] Implement. Run the tests; they pass.

### Task 4: Attendance features

**Files:**
- Create: `dropout_v2/attendance.py`, `dropout_v2/tests/test_attendance.py`

**Interfaces:**
- Produces:
  - `attendance.month_arrays(df, months) -> tuple[np.ndarray, np.ndarray]` (attended and working days, shape n×k, NaN = no data)
  - `attendance.attendance_features(att, wd, months, prefix) -> dict[str, np.ndarray]`. Keys: `{prefix}att_rate`, `days_attended`, `working_days`, `months_reported`, `months_zero`, `trailing_zero_months`, `first_month_rate`, `last_month_rate`, `trend`, `last_att_month`, `q1_rate`..`q4_rate`, `any_attendance`

- [ ] Write tests:
  - rate = sum(att) / sum(wd) (not the mean of monthly rates)
  - trailing zero months count from the end of the window
  - `last_att_month` is 1-based, NaN when never attended
  - quarter rates use only visible months
  - `test_zero_working_days_is_no_data`: a month with wd=0 does not count as reported
  - empty `months` → all-NaN or zero arrays without error
- [ ] Implement. Run the tests; they pass.

### Task 5: Labels

**Files:**
- Create: `dropout_v2/labels.py`, `dropout_v2/tests/test_labels.py`

**Interfaces:**
- Consumes: cleaned frames (Task 3)
- Produces:
  - `labels.school_reporting_share(df, months) -> pd.Series` (indexed by `school_key`)
  - `labels.build_labels(base: pd.DataFrame, nxt: pd.DataFrame, window=LABEL_WINDOW, reporting_min=0.5) -> pd.DataFrame`
    - `base` needs `person_key`, `student_key`
    - returns `person_key`, `label` (float 0/1/NaN), `label_reason` (category) aligned to `base` rows
  - Reasons: `CONTINUED_ATTENDING`, `CONTINUED_CLASS_11_PLUS`, `DROPOUT_NOT_FOUND`, `DROPOUT_NOT_ATTENDING`, `UNKNOWN_SCHOOL_NOT_REPORTING`

- [ ] Write one test per reason, plus:
  - `student_id` rescue (`person_id` changed, the same `student_key` has evidence → 0)
  - attendance only in August → not evidence for the Jun–Jul window
  - order: evidence wins over class 11
- [ ] Implement. Run the tests; they pass.

### Task 6: Features

**Files:**
- Create: `dropout_v2/features.py`, `dropout_v2/tests/test_features.py`

**Interfaces:**
- Consumes: Tasks 3–5, `reference.TRIBAL_MANDALS`, `reference.MANAGEMENT_GROUPS`
- Produces:
  - `features.build_features(year, snapshot, frames: dict[str, pd.DataFrame], schools: pd.DataFrame, prev_labels: pd.DataFrame | None) -> pd.DataFrame`
    - `frames` maps year → cleaned frame; years that are not in the data are absent
    - `prev_labels` = labels of base year T−1 with `school_key`, `mandal_key`
  - `features.feature_columns(snapshot) -> tuple[list[str], list[str]]` (numeric, categorical)
  - `features.ID_COLUMNS: list[str]` (person_key, person_id, student_key, school_key, academic_year, district, mandal, class_t, gender, social_category, mgmt_group, engagement_status)
  - `features.prepare_school_dim(schools: pd.DataFrame) -> pd.DataFrame`

- [ ] Write tests:
  - only classes 1–9; deceased excluded
  - `test_snapshot_features_ignore_future`: changing T's months after the snapshot, T+1 data, and later years leaves the features identical (for `jul` and `nov`)
  - `prev_status`: PROMOTED / REPEATED / OTHER_CHANGE / NOT_IN_PREV_YEAR / PREV_YEAR_DATA_GAP (2024-25 class 2 with the 2023-24 class 1 gap) / PREV_YEAR_NOT_AVAILABLE (2022-23)
  - `rejoiner_flag` is NaN when T−2 is absent
  - engagement statuses: ATTENDING / NOT_ATTENDING / NO_SCHOOL_DATA / UNKNOWN (`start`)
  - school rates are smoothed toward the global rate and are absent for `start`
  - `apr` has current-year marks columns; `jul` does not
  - `age_t` from DOB on 1 June of T; invalid age → NaN
- [ ] Implement. Run the tests; they pass.

### Task 7: DataStore and caching

**Files:**
- Create: `dropout_v2/dataset.py`, `dropout_v2/tests/test_dataset.py`

**Interfaces:**
- Produces:
  - `DataStore(tag: str, root: Path | None = None)` with:
    - `raw_path(year)`, `schools_path()`, `clean(year) -> pd.DataFrame | None` (cached; built from raw when missing)
    - `schools() -> pd.DataFrame`
    - `labels(year) -> pd.DataFrame | None` (base year labels; None when T+1 is absent)
    - `dataset(year, snapshot, with_label=True) -> pd.DataFrame`
    - `diagnostics: dict[str, dict]`
    - `years_available() -> list[str]`
  - Cache files: `clean/<year>.parquet`, `labels/<year>.parquet`, `datasets/<snapshot>/<year>.parquet`

- [ ] Write tests on a temp dir with the synthetic cache:
  - clean files are created and reloaded with the same dtypes
  - labels are None for 2026-27
  - the dataset for (2024-25, `jul`) has `label` and the feature columns
  - a missing raw file raises `FileNotFoundError` with an "extract" hint
- [ ] Implement. Run the tests; they pass.

### Task 8: Modeling

**Files:**
- Create: `dropout_v2/modeling.py`, `dropout_v2/tests/test_modeling.py`

**Interfaces:**
- Produces:
  - `fit_categories(df, categorical, min_count) -> dict[str, list[str]]`
  - `prepare_matrix(df, numeric, categorical, categories) -> pd.DataFrame`
  - `negative_sample(keys: np.ndarray, y: np.ndarray, neg_per_pos, rate_min) -> tuple[np.ndarray, np.ndarray, float]` (mask, weights, rate)
  - `train_with_early_stopping(Xtr, ytr, wtr, Xva, yva, params, max_rounds, patience) -> tuple[lgb.Booster, int]`
  - `train_fixed(X, y, w, params, n_rounds) -> lgb.Booster`
  - `fit_calibrator(raw, y) -> IsotonicRegression`
  - `ModelBundle` dataclass: `snapshot`, `numeric`, `categorical`, `categories`, `booster`, `best_iteration`, `calibrator`, `neg_rate`, `train_years`, `calibration_year`, `params`
    - `.predict_raw(df)`, `.predict(df)` (calibrated)
  - `save_bundle(bundle, path)`, `load_bundle(path) -> ModelBundle`

- [ ] Write tests:
  - `test_unseen_category_maps_to_other`
  - negative sampling keeps all positives, is deterministic, and its weights restore the negative count (± 5%)
  - early stopping returns best_iter ≥ 1
  - calibrated predictions are in [0, 1] and monotonic in the raw score
  - save/load round-trip gives identical predictions
- [ ] Implement. Run the tests; they pass.

### Task 9: Evaluation and explanations

**Files:**
- Create: `dropout_v2/evaluate.py`, `dropout_v2/explain.py`, `dropout_v2/tests/test_evaluate.py`, `dropout_v2/tests/test_explain.py`

**Interfaces:**
- Produces:
  - `evaluate.core_metrics(y, p) -> dict` (n, positives, positive_rate, auc, ap, brier)
  - `evaluate.topk_table(y, p, ks) -> DataFrame`
  - `evaluate.calibration_table(y, p, bins=10) -> DataFrame`
  - `evaluate.breakdown(df, y_col, p_col, by, k=0.10) -> DataFrame`
  - `evaluate.bootstrap_ci(y, p, reps, max_rows, seed=0) -> dict`
  - `evaluate.single_feature_auc(X, y, max_rows=500_000) -> DataFrame`
  - `evaluate.rule_baseline(df, snapshot) -> np.ndarray`
  - `evaluate.logistic_baseline(train_df, y_train, test_df, numeric, categorical, sample_rows) -> np.ndarray`
  - `evaluate.plot_pr_curve(...)`, `plot_calibration(...)`, `plot_importance(...)` → PNG paths
  - `explain.contributions(booster, X, num_iteration, chunk) -> np.ndarray`
  - `explain.top_reasons(contrib, X_values: pd.DataFrame, k=3) -> pd.DataFrame` (reason_1..3)
  - `explain.global_importance(booster, X_sample, num_iteration) -> DataFrame`

- [ ] Write tests:
  - top-k precision and recall are correct on a hand-made example
  - `test_breakdown_handles_single_class_group` → NaN AUC, no exception
  - bootstrap CI low ≤ point ≤ high
  - reasons pick the positive contributions in descending order, with readable text for `cur_att_rate` and `prev_status`
- [ ] Implement. Run the tests; they pass.

### Task 10: Report and training CLI (end-to-end on synthetic data)

**Files:**
- Create: `dropout_v2/report.py`, `dropout_v2/train.py`, `dropout_v2/tests/test_end_to_end.py`

**Interfaces:**
- Produces:
  - `train.run_training(tag, snapshots, tune=0, refit=True, include_2022=False, bootstrap_reps=100, baselines=True, run_dir=None) -> Path` (run dir)
  - CLI `python -m dropout_v2.train [--sample-pct N | --synthetic | --tag T] [--snapshots a,b] [--tune N] [--no-refit] [--include-2022] [--bootstrap N] [--no-baselines]`
  - Run dir: `config.json`, `run.log`, `data_profile.md`, `label_report.csv`, `bundle_<snap>.joblib`, `model_<snap>.txt`, `metrics_<snap>.json`, `topk_<snap>.csv`, `breakdown_<snap>_<by>.csv`, `importance_<snap>.csv`, PNGs, `MODEL_REPORT.md`; updates `artifacts/v2/LATEST`

- [ ] Write the end-to-end test: synthetic cache in a temp dir → `run_training(snapshots=["start", "jul", "apr"], bootstrap_reps=5)`. Check:
  - the files exist
  - test AUC for `jul` > 0.7
  - the report mentions every snapshot
- [ ] Implement. Run the tests; they pass.

### Task 11: Database access and extraction CLI

**Files:**
- Create: `dropout_v2/db.py`, `dropout_v2/extract.py`, `dropout_v2/tests/test_extract.py`

**Interfaces:**
- Produces:
  - `db.connect()`
  - `db.rows_to_table(rows, schema) -> pa.Table`
  - `db.fetch_to_parquet(query, out_path, schema, batch_size, connect_fn) -> int`
  - `extract.RAW_SCHEMA: pa.Schema`, `extract.SCHOOL_SCHEMA`
  - `extract.build_year_query(year, sample_pct) -> str`, `extract.build_school_query() -> str`
  - `extract.run_extract(years, sample_pct, refresh, batch_size) -> dict`
  - CLI `python -m dropout_v2.extract [--years ...] [--sample-pct N] [--refresh] [--batch-size N]`

- [ ] Write tests:
  - the query contains the year filter, the class range, the sample predicate only when pct > 0, and every `RAW_SCHEMA` column
  - `test_rows_to_table_coerces_types` (Decimal, date, bytes, None, numeric strings)
  - `fetch_to_parquet` with a fake cursor writes all batches and raises on a column-name mismatch
  - `RAW_SCHEMA` names == `synthetic.RAW_COLUMNS`
- [ ] Implement. Run the tests; they pass.

### Task 12: Scoring CLI

**Files:**
- Create: `dropout_v2/score.py`, `dropout_v2/tests/test_score.py`

**Interfaces:**
- Produces:
  - `score.detect_available_months(df, min_coverage=0.5) -> int`
  - `score.choose_snapshot(available_months, trained: list[str]) -> str`
  - `score.assign_tiers(df, prob_col) -> pd.DataFrame` (adds `risk_tier`, `state_pct`, `district_pct`)
  - `score.run_scoring(tag, run_dir=None, snapshot="auto", live_year=LIVE_YEAR, legacy_export=False) -> Path`
  - CLI `python -m dropout_v2.score [--tag/--sample-pct/--synthetic] [--run DIR] [--snapshot auto|name] [--legacy-export]`

- [ ] Write tests:
  - `test_detect_available_months_ignores_zero_months` (Aug–May 0/0 → 2)
  - `choose_snapshot(2, all) == "jul"`, `choose_snapshot(1, all) == "start"`
  - tier shares match the cut-offs among ATTENDING students; NOT_ATTENDING keeps its own tier
  - the end-to-end score on synthetic data writes the parquet, `district_summary.csv`, `top_risk_by_district.csv`, `not_attending.csv` with the expected columns
  - the legacy export has the columns used by `app/views/1_Student_Risk_Scores.py`
- [ ] Implement. Run the tests; they pass.

### Task 13: Runbook and final verification

**Files:**
- Create: `docs/dropout_v2_RUNBOOK.md`

- [ ] Write the runbook:
  - copy the files to the server
  - `selftest`
  - sample extract → sample train → full extract → train `jul` → score → train all snapshots
  - what to send back; troubleshooting
- [ ] Run the full test suite in the Python 3.9 venv. Run the synthetic CLI flow: `extract`-free `train --synthetic` → `score --synthetic`.
- [ ] Review the whole diff against the spec. Check:
  - leakage rules
  - class 10 exclusion
  - Python 3.9 syntax (`python -m py_compile` under 3.9)
