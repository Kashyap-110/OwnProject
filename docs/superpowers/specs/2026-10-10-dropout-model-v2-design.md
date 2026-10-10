# Student Dropout Early-Warning Model v2 — Design Spec

| Item | Value |
|---|---|
| Date | 2026-10-10 |
| Status | Draft for review |
| Source table | `aiml_usecase_curated.final_student_dropout_info_new` (load_date 2026-10-08) |
| Decision | Option A — quarterly early warning |
| Deadline | Demo to officials on Monday 2026-10-12 |
| Runtime | Offline Linux server: Python 3.9.25, 16 CPUs, 49 GB RAM, packages as in `requirements.txt` (pandas 2.3.3, numpy 2.0.2, scikit-learn 1.6.1, lightgbm 4.6.0, pyarrow 21.0.0, shap 0.49.1, matplotlib 3.9.4). No pytest. |

---

## 1. Understanding

**What the user said**
- Build the dropout model for classes 1–10 of AP Government schools, for production use.
- Use option A: risk lists that refresh during the year as attendance arrives.
- The data team needs more than one week for the staging-table changes. The model must be built now, with all data refinement done in code, without changing the source table.
- The code runs on the offline server; the user runs it and reports results. No new packages.

**Assumptions (correct me)**
- The Monday demo needs: honest out-of-time metrics, a live risk list for 2026-27, district summaries, and plain-language reasons per student. A model report (Markdown + charts + CSVs) is the main demo artefact.
- The Streamlit app and FastAPI are migrated later (separate spec). For the demo, an optional export lets the existing "Student Risk Scores" page show the new scores.
- This version is a pilot ("version 1"). It is not used for official dropout statistics. Version 2 is trained after the data team's fixes arrive.

**Success criteria**
1. One command extracts data, one trains, one scores; all run on the server without edits.
2. No feature uses information from after its snapshot date (enforced by an automated test).
3. Metrics are measured on a later year than training (2025-26), with confidence intervals, overall and for students still attending.
4. Live scores for every 2026-27 student in classes 1–10, with tier, district rank, and top-3 reasons.
5. The user can do a quick end-to-end dry run on a 2% sample before the full run.

---

## 2. Scope

**In scope**
- New package `dropout_v2/` (the old `ml_pipeline/` stays unchanged).
- Extraction from Impala to a local Parquet cache.
- Cleaning, de-duplication, label construction, time-safe features for 6 snapshots.
- LightGBM training with early stopping, calibration, out-of-time evaluation, baselines, explanations.
- Live scoring of 2026-27, district summaries, top lists, model report.
- Unit tests (unittest) with a synthetic data generator; a `selftest` command for the server.

**Out of scope**
- Changes to the source table (data team, documents R01–R46).
- Dashboard and API migration; the manual weight matrix.
- CatBoost/XGBoost challengers, sibling/household-network features, fairness mitigation (fairness is reported, not corrected).

---

## 3. Key decisions

### 3.1 Label: what "dropout" means in this version

The audit showed that the source system keeps students in the table after they leave. "No row next year" finds only 0.09–0.64% of students for recent years. So the label uses **evidence of attendance in the next year**.

For a base row (student *p*, academic year *T*, class 1–10):

| Case (checked in this order) | Label | `label_reason` |
|---|---|---|
| A matched row in *T+1* has attendance > 0 in the label window | 0 | `CONTINUED_ATTENDING` |
| A matched row in *T+1* is in class ≥ 11 | 0 | `CONTINUED_CLASS_11_PLUS` |
| No matched row in *T+1* | 1 | `DROPOUT_NOT_FOUND` |
| Matched row(s) exist, none has attendance in the window, and the school of the row reports attendance | 1 | `DROPOUT_NOT_ATTENDING` |
| Matched row(s) exist, none has attendance, and the school does not report attendance | excluded | `UNKNOWN_SCHOOL_NOT_REPORTING` |

Definitions:
- **Matched row:** a *T+1* row with the same `person_id`; or, when that row has no attendance, a *T+1* row with the same `student_id` as the base row. The `student_id` match rescues children whose `person_id` changed (68,088 such `student_id` values in the audit).
- **Label window:** months of *T+1* that count as evidence. Default `["jun", "jul"]`, because 2026-27 has only June and July loaded. The same window is used for every year, so all labels mean the same thing. When August–September 2026 arrive, the window can grow in config.
- **School reports attendance:** at least 50% of that school's *T+1* rows have working days > 0 in the label window.
- **Class 11+ rule:** class 11/12 rows come from junior colleges, whose attendance coverage is unknown. A class 11+ row counts as continuation.

Known noise that stays (documented in the report): transfers to schools outside the table, migration out of AP, and class 10 students who join ITI/polytechnic count as dropouts.

### 3.2 Base population and exclusions

A base row is a de-duplicated student-year row with class 1–10. Excluded:
1. `person_id` that is empty or not numeric.
2. **Placeholder IDs:** a `person_id` with more than 3 rows in one year (the 127,571-row ID in 2024-25). All rows of that ID in that year are excluded.
3. Students with `is_deceased = "Deceased"` (any row). "reported deceased" is kept, because 96.8% of those students still attend.
4. Rows with label `UNKNOWN_SCHOOL_NOT_REPORTING` (excluded from training and evaluation only; they are still scored live).

### 3.3 Snapshots (option A)

A snapshot is the point in year *T* when the risk list is made. A snapshot model sees only the months of *T* up to its cut-off.

| Snapshot | Current-year months visible | Use |
|---|---|---|
| `start` | none (only prior year + profile) | June, before attendance arrives |
| `jul` | Jun–Jul | **Demo now** (2026-27 data stops at July) |
| `aug` | Jun–Aug | Quarter 1 |
| `nov` | Jun–Nov | Quarter 2 |
| `feb` | Jun–Feb | Quarter 3 |
| `apr` | Jun–Apr + current-year marks | Quarter 4 / end of year |

- One LightGBM model per snapshot, all from the same code.
- At scoring time, the code finds the last month of the live year with attendance for at least 50% of rows. It then uses the latest snapshot whose months are all available.
- Current-year marks (`fa_marks`, `sa_marks`) are year totals, so only the `apr` snapshot uses them. Earlier snapshots use prior-year marks.

### 3.4 Years and split

| Base year *T* | Label from | Role |
|---|---|---|
| 2022-23 | 2023-24 | History only. Not used for training, because 74.9% of its "not found" students reappear in 2024-25 (missing 2023-24 rows). Config flag can include it. |
| 2023-24 | 2024-25 | **Train** |
| 2024-25 | 2025-26 | **Validation**: early stopping, calibration, tier calibration |
| 2025-26 | 2026-27 (Jun–Jul) | **Test**: untouched; all reported metrics |
| 2026-27 | — | **Live scoring** |

- **Evaluation model:** train on 2023-24, early stopping on 2024-25, calibrate on 2024-25, test on 2025-26.
- **Production model** (used for live scoring): retrain on 2023-24 + 2024-25 with the evaluation model's best number of trees × 1.15, then calibrate on 2025-26. The reported metrics always come from the evaluation model.

### 3.5 Model

- **LightGBM** (binary). It is fast on millions of rows, handles missing values natively, and stores the category mapping. Defaults: learning rate 0.05, `num_leaves` 127, `min_data_in_leaf` 500, `feature_fraction` 0.8, bagging 0.8, `lambda_l2` 1.0, at most 3,000 rounds, early stopping after 100 rounds without improvement in average precision.
- **Imbalance:** keep all positives and a deterministic sample of negatives (target 5 negatives per positive, rate between 0.05 and 1.0). Each sampled negative gets weight 1/rate. No other class weighting.
- **Early stopping** runs on a stratified 1.5M-row sample of the validation year, for speed. Calibration and metrics use the full year.
- **Calibration:** isotonic regression on validation predictions.
- **Optional tuning** (`--tune N`): random search over `num_leaves`, `min_data_in_leaf`, `feature_fraction`, `lambda_l2`, on a 30% sample. Off by default.
- **Categoricals:** each categorical column gets a fixed category list from training, stored with the model. Unseen values become `OTHER`. This removes the category-code bug of the old pipeline.

### 3.6 Tiers and outputs

Engagement status is computed for every live student from the visible months:
- `ATTENDING`: attended at least 1 day.
- `NOT_ATTENDING`: the school reports attendance, but the student has none.
- `NO_SCHOOL_DATA`: the school does not report attendance.
- `UNKNOWN`: the `start` snapshot, which has no current-year data.

Risk tiers are given among students who are `ATTENDING` or `UNKNOWN`. Cut-offs by statewide rank (configurable):
- `CRITICAL`: top 2%
- `HIGH`: next 5%
- `MEDIUM`: next 10%
- `LOW`: the rest

`NOT_ATTENDING` and `NO_SCHOOL_DATA` students form separate lists, ordered by probability. The report shows the precision of each tier measured on the test year.

---

## 4. Data refinement rules (in code)

| Problem (audit) | Rule |
|---|---|
| IDs with leading zeros / mixed types | `person_key` = int64 of the digits of `person_id` (used for joins); the original string is kept for output. `student_id`, `school_id` → int64. |
| Duplicate rows per student-year | Keep one main row: most attended days in the year; ties → higher class → higher `school_id`. Monthly attendance = maximum per month across the student's rows. Keep `n_rows_in_year`. |
| Placeholder `person_id` | Excluded (3.2). |
| Class codes −3..0, 13 | Keep −3..12 for history and labels. Drop 0 and 13. Base rows are 1–10. |
| `student_age` is today's age | Not used. Age on 1 June of *T* comes from the date of birth. Ages < 3 or > 25 → missing. `over_age = age − (class + 5)`. |
| `highest_qualification`, `is_dropout`, `dropout_reason` | Never extracted as features. `is_dropout` is extracted only for a comparison table in the report. |
| Snapshot columns that change after a dropout (marital status, members count, siblings, health, `child_status`) | Not used. |
| Columns with a meaning change between years (`thalliki_vandanam`, kits) | Not used in this version. |
| `income_band` (bad encoding, BPL conflict) | Not used. |
| NULL tokens (`'null'`, `''`, `'-'`, `'NA'`, …), extra spaces, case | Text is trimmed and upper-cased; tokens → `UNKNOWN`. Rare categories (< 1,000 rows in training) → `OTHER`. |
| School management (≈40 values) | Mapped to 11 groups: `GOVT_LOCAL_BODY`, `GOVT_STATE`, `RESIDENTIAL_WELFARE`, `PRIVATE_UNAIDED_STATE`, `PRIVATE_UNAIDED_CBSE_ICSE`, `PRIVATE_AIDED`, `SPECIAL_NEEDS`, `MADARSA_ORIENTAL_SANSKRIT`, `CENTRAL`, `NCLP_UNRECOGNISED`, `OTHER`. |
| Attendance anomalies | Attended is clipped to [0, working days]. Working days > 31 → missing. Missing or 0 working days → "no data" for that month. |
| Marks scale and impossible values | For each year × class: values > 3 × the 99th percentile → missing. Marks ≤ 0 → zero flag. The feature is the percentile rank within year × class × board group, among positive marks. |
| Coordinates | Outside latitude 12–20 / longitude 76–85 → missing. Home–school distance > 50 km → missing. |
| Teachers = 0 in private schools | 0 → missing; pupil–teacher ratio only when teachers > 0. |
| 2023-24 rows missing for class 1 and pre-primary | Known-gap rule: when the expected prior-year class (class − 1) falls in a known gap of *T−1*, prior-year status = `PREV_YEAR_DATA_GAP`, not "new". Config: `KNOWN_GAPS = {"2023-24": [-3, -2, -1, 1]}`. |
| 2022-23 has no June–July attendance | Prior-year features use annual and last-quarter rates only, not prior-year quarter 1. |

---

## 5. Features

All features are computed for base year *T* and one snapshot. "Prior year" = *T−1*.

**Profile (known at the start of *T*)**
- `class_t`, `age_t`, `over_age`, `dob_missing`, `gender`
- `social_category` (toggle `USE_SOCIAL_CATEGORY`, default on; religion and caste are never used)
- `rural_urban`, `ration_card`, `dwelling`, `has_father_id`, `has_mother_id`, `medium`
- `district`, `is_tribal_mandal` (existing list from `ml_pipeline/preprocess.py`), `home_school_km`

**School (*T* row's school)**
- `mgmt_group`, `school_category`, `board_group`, `school_gender_type`, `school_rural_urban`
- `school_rows_t` (enrolment counted from the table for year *T*), `school_size_growth` (rows *T* / rows *T−1*), `school_new_in_t`
- `ptr` (pupil–teacher ratio, government schools)
- `is_terminal_class` (class = school's highest class, so the student must change school)

**Prior year**
- `prev_status`: `PROMOTED` / `REPEATED` / `OTHER_CHANGE` / `NOT_IN_PREV_YEAR` / `PREV_YEAR_DATA_GAP` / `PREV_YEAR_NOT_AVAILABLE`
- `prev_same_school`, `rejoiner_flag` (absent in *T−1*, present in *T−2*; missing when *T−2* is not in the data)
- `prev_att_rate`, `prev_last_q_att_rate` (Mar–Apr), `prev_months_reported`, `prev_months_zero`, `prev_last_att_month`, `prev_no_attendance`, `prev_n_rows`
- `prev_fa_rank`, `prev_sa_rank`, `prev_fa_zero`, `prev_sa_zero`, `prev_marks_missing`

**Current year up to the snapshot** (not in `start`)
- `cur_att_rate`, `cur_days_attended`, `cur_working_days`, `cur_months_reported`, `cur_months_zero`, `cur_trailing_zero_months`
- `cur_first_month_rate`, `cur_last_month_rate`, `cur_trend`
- Quarter rates `cur_q1_rate`…`cur_q4_rate` (visible months only)
- `engagement_status`
- `school_cur_reporting_share`, `school_cur_att_rate`, `att_vs_school`

**Past outcomes of school and mandal** (not in `start`, because the *T−1* → *T* labels need June–July of *T*)
- `school_prev_dropout_rate`, `mandal_prev_dropout_rate`: smoothed with prior weight 50 toward the state rate, from base-year *T−1* labels.

**End of year only** (`apr`)
- `cur_fa_rank`, `cur_sa_rank`, `cur_fa_zero`, `cur_sa_zero`, `cur_marks_missing`, `n_rows_in_year`

**Time-safety rule:** a feature for snapshot *s* may use only:
- profile columns;
- year *T* months up to *s*;
- years *T−1* and *T−2*;
- *T−1* labels (only from `jul` onward).

A unit test changes all later months and all *T+1* data, then checks that the features do not change.

---

## 6. Architecture

```
dropout_v2/
  config.py      constants: table, years, months, snapshots, label window, exclusions, maps, model params, paths
  db.py          Impala connection (raises on error) and streaming fetch into Parquet
  extract.py     CLI: per-year extraction into data/v2/<tag>/raw/<year>.parquet (+ school dimension)
  clean.py       raw year -> cleaned student-year frame (types, IDs, dedup, placeholders, text maps, attendance, marks)
  attendance.py  pure functions: monthly arrays + visible months -> attendance features
  labels.py      base year T + cleaned T+1 -> label, label_reason, diagnostics
  features.py    base year T + snapshot -> feature frame (profile, school, prior year, current year, rates)
  dataset.py     builds and caches data/v2/<tag>/datasets/<snapshot>/<year>.parquet
  modeling.py    sampling, LightGBM fit, calibration, category handling, model bundle save/load
  evaluate.py    metrics, top-k, calibration table, breakdowns, bootstrap CIs, baselines, plots
  explain.py     pred_contrib -> top-3 reasons; global importance
  report.py      Markdown model report
  train.py       CLI: datasets -> evaluation model -> test metrics -> production model -> artefacts
  score.py       CLI: live year -> snapshot auto-select -> scores, summaries, top lists, optional legacy export
  selftest.py    CLI: runs the unit tests on the server
  synthetic.py   synthetic raw data with planted patterns (tests and dry runs without the database)
  tests/         unittest modules
```

**Data flow**
1. `extract` → raw Parquet per year (all classes −3..12, selected columns only), plus a school dimension table.
2. `clean` → cleaned frame per year (cached).
3. `dataset`: for each base year *T* and snapshot → features + label (cached).
4. `train` → artefacts in `artifacts/v2/<run_id>/`, and `artifacts/v2/LATEST` names the newest run.
5. `score` → scores and summaries in `artifacts/v2/<run_id>/scores/`.

**Extraction**
- One `SELECT` per year with explicit `CAST`s, streamed with `fetchmany(200_000)` into a `ParquetWriter` with a fixed schema. This avoids the old type-drift problem.
- Parent IDs become 0/1 flags in SQL.
- School text attributes come from a separate one-row-per-school query.
- **Sample mode** (`--sample-pct 2`) uses `ABS(fnv_hash(person_id)) % 100 < 2`, so the same students are sampled in every year.

**Run directory contents**
- `config.json`, `run.log`
- `data_profile.md`, `label_report.csv`
- for each snapshot: `model_<snapshot>.txt` (LightGBM), `bundle_<snapshot>.joblib` (features, categories, calibrator, best iteration, thresholds), `metrics_<snapshot>.json`, breakdown CSVs, `importance_<snapshot>.csv/.png`, `calibration_<snapshot>.png`, `pr_curve_<snapshot>.png`
- `MODEL_REPORT.md`

**Live outputs (`scores/`)**
- `scores_<year>_<snapshot>.parquet` with: IDs, class, gender, district, mandal, school, management group, engagement status, calibrated probability, state and district percentiles, tier, 3 reasons, key feature values.
- `district_summary.csv`
- `top_risk_by_district.csv` (top 500 attending students per district)
- `not_attending.csv`
- optional `--legacy-export`, which writes `artifacts/statewide_scores.parquet` with the column names that `app/views/1_Student_Risk_Scores.py` expects.

---

## 7. Evaluation (reported for each snapshot, on test year 2025-26)

- ROC-AUC, average precision (PR-AUC), Brier score, positive rate.
- Precision, recall, and lift at the top 1, 2, 5, 10, and 20% of students.
- A calibration table (10 bins) and plot.
- Breakdowns by class, district, gender, social category, management group, and engagement status: n, positive rate, AUC, AP, recall at the top 10%.
- **Separate metrics for `ATTENDING` students:** the honest early-warning measure.
- 95% bootstrap confidence intervals (100 resamples, up to 2M rows) for AUC, AP, and recall at the top 5%.
- **Baselines on the same test rows:**
  - an attendance rule (1 − current attendance rate; 1 − prior attendance rate for `start`);
  - logistic regression on a 1M-row training sample.
- **Leakage alarm:** single-feature AUC on validation for every numeric feature. Anything above 0.95 is flagged in the report.
- **Comparison table:** label versus the source `is_dropout` flag and versus "not found next year".

---

## 8. Error handling and logging

- Database errors raise with the query name; nothing returns an empty frame silently.
- After extraction: expected columns exist, and each year has more than 0 rows. Otherwise the run stops with a clear message.
- Sanity warnings in the log and the report:
  - label positive rate outside 0.5–30%;
  - unknown-label share above 30%;
  - fewer than 1,000 positives in any split;
  - live-year attendance coverage below 50%.
- The log goes to the console and to `run.log`, with step timings and peak memory (via `resource` on Linux).
- Each CLI checks its config at start (years exist, snapshots known, label window valid).

---

## 9. Testing

- **Local:** a `uv` virtual environment with Python 3.9 and the server's exact package versions.
- **`synthetic.py`** generates raw yearly frames with the real column names and planted cases: carried-forward rows with no attendance, a placeholder ID, duplicate rows, deceased students, a non-reporting school, class 10 → 11, `student_id` rescue, NULL tokens, marks outliers, and a dropout signal tied to falling attendance.
- **Unit tests:**
  - cleaning rules;
  - every label case;
  - snapshot masking (the leakage test);
  - prior-year status and the known-gap rule;
  - marks ranks;
  - school aggregates;
  - category handling for unseen values;
  - SQL builder;
  - batch-to-Arrow conversion with a fake cursor.
- **End-to-end tests:**
  - synthetic data → train (all snapshots, small) → score; artefacts exist, and AUC > 0.7 on the planted signal;
  - scoring output columns and tiers.
- **On the server:**
  - `python -m dropout_v2.selftest` runs the same tests;
  - the dry run on the 2% sample;
  - then the full run.

---

## 10. How the user runs it (server)

```bash
python -m dropout_v2.selftest
python -m dropout_v2.extract --sample-pct 2
python -m dropout_v2.train --sample-pct 2 --snapshots jul
python -m dropout_v2.extract
python -m dropout_v2.train --snapshots jul
python -m dropout_v2.score --snapshot auto
python -m dropout_v2.train --snapshots start,jul,aug,nov,feb,apr
```

Estimated full-data times on 16 cores (to be confirmed on the server):
- extraction: 30–60 minutes once (then cached);
- training one snapshot: 15–30 minutes;
- scoring: 10–20 minutes.

---

## 11. Known limitations (stated in the model report)

1. The label is a proxy. It counts transfers out of the table, migration out of AP, and ITI/polytechnic joiners after class 10 as dropouts.
2. Test-year labels use only June–July 2026.
3. The evaluation model has one training year (2023-24), because 2022-23 labels are unreliable.
4. Slow-changing household and school values are current snapshots, so there is a small leakage risk.
5. 127,571 students of 2024-25 with the placeholder ID are excluded.
6. Pilot only. Do not use for official dropout statistics. Retrain when the data team's changes R03, R08, R12, R13, R16, and R17 arrive.
