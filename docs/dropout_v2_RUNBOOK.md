# Dropout Model v2 — Server Runbook

This guide runs the v2 pipeline on the offline server. It uses only the packages that are already installed (`requirements.txt`). Every command runs from the project folder (`student_dropout_system_final`) with your virtual environment active.

```bash
cd ~/student_dropout_system_final
source stud_venv/bin/activate
```

## 0. Copy the code

Copy the folder `dropout_v2/` (and `docs/` if you want the documents) into the project folder on the server. The old `ml_pipeline/`, `api/` and `app/` folders do not change.

## 1. Check the server (2–4 minutes)

```bash
python -m dropout_v2.selftest
```

Expected last line: `SELFTEST PASSED: 117 tests`. If a test fails, send me the full output.

## 2. Dry run on a 2% sample (15–30 minutes)

The sample contains the same 2% of students in every year. It proves that the whole pipeline works on the real table before the long run.

```bash
klist
python -m dropout_v2.extract --sample-pct 2
python -m dropout_v2.train --sample-pct 2 --snapshots jul
python -m dropout_v2.score --sample-pct 2
```

- `klist` must show a valid Kerberos ticket. If it does not, run `kinit` first.
- `extract` writes `data/v2/sample2/raw/<year>.parquet` and `schools.parquet`.
- `train` prints the run folder: `artifacts/v2/<date>_sample2/`.
- `score` writes `artifacts/v2/<date>_sample2/scores/`.

Send me these files from the run folder:
- `MODEL_REPORT.md`
- `label_summary.csv`
- `data_profile.md`
- `single_feature_auc_jul.csv`
- `run.log` (only if something failed)

## 3. Full extraction (30–90 minutes, once)

Long commands can stop when the SSH session closes, so run them with `nohup`:

```bash
nohup python -m dropout_v2.extract > extract.log 2>&1 &
tail -f extract.log
```

Files appear in `data/v2/full/raw/`. A year that is already extracted is skipped. Use `--refresh` to extract it again.

Disk space: about 1 GB per year for raw files, plus cleaned files and datasets. Check free space first with `df -h .`; keep at least 30 GB free.

## 4. Train the demo model (`jul` snapshot, 20–60 minutes)

```bash
nohup python -m dropout_v2.train --snapshots jul > train_jul.log 2>&1 &
tail -f train_jul.log
```

The log shows each step with its duration and peak memory. At the end:
`Training finished. Report: artifacts/v2/<date>_full/MODEL_REPORT.md`

## 5. Score the live year 2026-27 (10–30 minutes)

```bash
python -m dropout_v2.score
```

The scorer finds how many months of 2026-27 attendance have arrived (now June–July) and selects the `jul` model. The outputs go to `artifacts/v2/<run>/scores/`:

| File | Content |
|---|---|
| `SCORING_SUMMARY.md` | Students by tier, district table |
| `scores_2026-27_jul.parquet` | Every student in classes 1–9: probability, tier, state and district rank, 3 reasons |
| `district_summary.csv` | Students and expected dropouts per district |
| `top_risk_by_district.csv` | Top 500 attending students per district, with reasons |
| `not_attending.csv` | Students with no attendance in the latest month at schools that report attendance |

To show the new scores on the existing Streamlit page "Student Risk Scores":

```bash
python -m dropout_v2.score --legacy-export
```

This writes `artifacts/statewide_scores.parquet`, and keeps a backup of the old file. The other dashboard pages still read the old table; they change in a later phase.

## 6. What to show in the demo

1. `MODEL_REPORT.md`, section 1. Point out:
   - test AUC and AP against the "rule AUC" (simple low-attendance rule);
   - recall at the top 10%;
   - the attending-only AUC.
2. `pr_curve_jul.png` and `calibration_jul.png`.
3. The tier table: the observed dropout rate in each tier on 2025-26, a year the model never saw.
4. `district_summary.csv` and one district from `top_risk_by_district.csv`, with the reasons.
5. The limits (end of `MODEL_REPORT.md`):
   - this is a pilot;
   - the label is a proxy;
   - data-team fixes are pending.

## 7. After the demo

| Task | Command |
|---|---|
| Train all quarterly snapshots | `nohup python -m dropout_v2.train > train_all.log 2>&1 &` |
| New month of attendance arrives | `python -m dropout_v2.extract --years 2026-27 --refresh`, then `python -m dropout_v2.score` |
| Hyperparameter search (slower) | `python -m dropout_v2.train --snapshots jul --tune 8` |
| Skip the logistic baseline (faster) | add `--no-baselines` |
| Try 2022-23 as an extra training year | add `--include-2022` |

When the label window should grow (for example, when August–September 2026 attendance has arrived), change `LABEL_WINDOW` in `dropout_v2/config.py`. Then train again. The code rebuilds labels and datasets automatically, because the cache folder name includes a hash of the rules.

## 8. Troubleshooting

| Message | Action |
|---|---|
| `Raw data for <year> not found ... Run: python -m dropout_v2.extract` | Run the extract command with the same `--sample-pct`. |
| `Query columns do not match the expected schema` | A column was renamed in the table. Send me the message. |
| Kerberos / GSSAPI error | Run `kinit`, then `klist`, then retry. |
| `MemoryError` or the process is killed | Close other jobs. Send me the log; the pipeline can process years in smaller pieces. |
| `No dropouts in the ... data` | The label found no dropouts. Send me `label_summary.csv`. |
| `ALERT: these features alone separate the classes` in the report | Possible leakage. Send me `single_feature_auc_<snapshot>.csv` before you use the model. |
| Old cached data after a code change | Delete `data/v2/<tag>/derived/` and run again. Raw files stay. |

Impala settings come from `dropout_v2/config.py`. You can override them with the environment variables `DROPOUT_IMPALA_HOST`, `DROPOUT_IMPALA_PORT`, `DROPOUT_IMPALA_AUTH`, and `DROPOUT_IMPALA_KERBEROS_SERVICE`.
