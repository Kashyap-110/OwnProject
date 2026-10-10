"""Markdown reports: MODEL_REPORT.md (training) and SCORING_SUMMARY.md (live scoring)."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from dropout_v2 import config
from dropout_v2.utils import md_table

LIMITATIONS = [
    "The label is a proxy: a student is a dropout when there is no attendance in the label window of the next year "
    "at a school that reports attendance, or when the student is not in the next year's records. Transfers to schools "
    "outside the table and migration out of Andhra Pradesh therefore count as dropouts.",
    "Students whose next-year school sends no attendance are excluded from training and evaluation (label unknown).",
    "Test-year labels use only the label-window months of the next year ({}).".format(", ".join(config.LABEL_WINDOW)),
    "The evaluation model has one training year (2023-24); 2022-23 labels are unreliable because many 2023-24 rows "
    "are missing in the source table.",
    "Household and school values (ration card, dwelling, school profile) are current snapshots, not year values. "
    "Columns known to change after a dropout (marital status, health, child status) are not used.",
    "Students with the placeholder person_id (127,571 rows in 2024-25) are excluded.",
    "Class 10 has no risk score in this version.",
    "Pilot model. Do not use for official dropout statistics. Retrain after the data team's changes "
    "R03, R08, R12, R13, R16 and R17.",
]


def _fmt(value, pct: bool = False) -> str:
    if value is None:
        return "n/a"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if v != v:
        return "n/a"
    return "{:.1%}".format(v) if pct else "{:.3f}".format(v)


def _topk_value(table: Optional[pd.DataFrame], k: float, col: str):
    if table is None or len(table) == 0:
        return None
    row = table.loc[(table["top_k"] - k).abs() < 1e-9]
    return None if row.empty else row.iloc[0][col]


def write_model_report(run_dir: Path, context: dict, results: Dict[str, dict]) -> Path:
    lines = ["# Student Dropout Early-Warning Model v2: Model Report", ""]
    lines += [
        "| Item | Value |", "|---|---|",
        "| Run | `{}` |".format(run_dir.name),
        "| Data tag | `{}` |".format(context.get("tag")),
        "| Created | {} |".format(context.get("created")),
        "| Train years | {} |".format(", ".join(context.get("train_years", []))),
        "| Validation year | {} |".format(context.get("valid_year")),
        "| Test year (untouched) | {} |".format(context.get("test_year")),
        "| Label window (months of next year) | {} |".format(", ".join(config.LABEL_WINDOW)),
        "| Classes scored | 1-9 (class 10 excluded) |",
        "| Rules signature | `{}` |".format(context.get("rules_signature")),
        "",
    ]

    lines += ["## 1. Summary (test year {})".format(context.get("test_year")), ""]
    summary = []
    for snap, res in results.items():
        m = res["metrics"]
        summary.append({
            "snapshot": snap,
            "test students": m["test"]["n"],
            "dropout rate": _fmt(m["test"]["positive_rate"], pct=True),
            "AUC": _fmt(m["test"]["auc"]),
            "AP (PR-AUC)": _fmt(m["test"]["ap"]),
            "recall @ top 10%": _fmt(_topk_value(res["topk"], 0.10, "recall"), pct=True),
            "precision @ top 5%": _fmt(_topk_value(res["topk"], 0.05, "precision"), pct=True),
            "AUC attending only": _fmt(m.get("attending", {}).get("auc")),
            "rule AUC": _fmt(m.get("rule_baseline", {}).get("auc")),
            "trees": m["best_iteration"],
        })
    lines.append(md_table(pd.DataFrame(summary)))
    lines += [
        "**How to read.** AUC: the chance that a random dropout gets a higher score than a random continuing "
        "student (0.5 = chance). AP: area under the precision-recall curve; compare it with the dropout rate. "
        "Recall @ top 10%: the share of real dropouts found when field staff visit the top 10% of students. "
        "Precision @ top 5%: the share of the top 5% who really drop out. "
        "'AUC attending only' excludes students already not attending, so it is the honest early-warning measure. "
        "'rule AUC' is a simple low-attendance rule; the model must beat it.",
        "",
    ]

    lines += ["## 2. Data quality (after cleaning)", ""]
    profile = context.get("profile")
    if profile is not None:
        lines.append(md_table(profile, floatfmt=".3f"))
    lines += ["`wd_<month>` = share of base students (classes 1-9) with attendance data for that month.", ""]

    lines += ["## 3. Labels", ""]
    lines += ["Dropout = no attendance in the label window of the next year at a reporting school, or not found "
              "next year. Students in non-reporting schools are 'unknown' and excluded from training.", ""]
    label_summary = context.get("label_summary")
    if label_summary is not None:
        lines.append(md_table(label_summary, floatfmt=".4f"))
    lines += ["`dropout_rate_if_src_true/false`: our label's dropout rate when the source `is_dropout` flag is "
              "TRUE / FALSE. A weak difference confirms that the source flag is unreliable.", ""]

    for i, (snap, res) in enumerate(results.items(), start=4):
        m = res["metrics"]
        lines += ["## {}. Snapshot `{}` ({} current-year months visible)".format(
            i, snap, len(config.visible_months(snap))), ""]
        lines += [
            "- Training rows: {:,} ({:,} after negative sampling at rate {:.3f}); validation rows: {:,}; "
            "test rows: {:,}.".format(m["n_train"], m["n_train_sampled"], m["neg_rate"], m["n_valid"], m["n_test"]),
            "- Trees: {} (evaluation model); production model refit: {}.".format(m["best_iteration"], m.get("refit")),
            "- Validation AUC {} / AP {}; test AUC {} / AP {}; test Brier {}.".format(
                _fmt(m["valid"]["auc"]), _fmt(m["valid"]["ap"]), _fmt(m["test"]["auc"]), _fmt(m["test"]["ap"]),
                _fmt(m["test"]["brier"])),
        ]
        ci = m.get("bootstrap_ci") or {}
        if ci:
            lines.append("- 95% bootstrap intervals (test): " + "; ".join(
                "{} [{}, {}]".format(k, _fmt(v[0]), _fmt(v[1])) for k, v in ci.items()))
        lines.append("")
        lines += ["**Top-k (test year, all students)**", "", md_table(res["topk"])]
        if res.get("topk_attending") is not None:
            lines += ["**Top-k (test year, attending students only)**", "", md_table(res["topk_attending"])]
        lines += ["**Risk tiers on the test year** (tiers among attending/unknown students; precision = observed "
                  "dropout rate in the tier)", "", md_table(res["tiers"])]
        baselines = []
        for name in ("model", "rule_baseline", "logistic_baseline"):
            core = m["test"] if name == "model" else m.get(name)
            if core:
                baselines.append({"scorer": name, "AUC": core.get("auc"), "AP": core.get("ap")})
        lines += ["**Model against baselines (test year)**", "", md_table(pd.DataFrame(baselines))]
        lines += ["**Calibration (test year)**", "", md_table(res["calibration"], floatfmt=".4f")]
        for by, table in res["breakdowns"].items():
            lines += ["**By {}**".format(by), "", md_table(table, floatfmt=".3f")]
        lines += ["**Top features (mean |contribution|)**", "", md_table(res["importance"].head(20), floatfmt=".4f")]
        alerts = res["single_auc"]
        lines += ["**Leakage check: single-feature AUC on validation (alert above {})**".format(
            config.SINGLE_FEATURE_AUC_ALERT), "", md_table(alerts.head(10), floatfmt=".3f")]
        flagged = alerts.loc[alerts["auc"] > config.SINGLE_FEATURE_AUC_ALERT, "feature"].tolist()
        if flagged:
            lines.append("> ALERT: these features alone separate the classes almost perfectly: {}. Check them for "
                         "leakage before use.".format(", ".join(flagged)))
        lines += ["", "Charts: `pr_curve_{0}.png`, `calibration_{0}.png`, `importance_{0}.png`.".format(snap), ""]

    lines += ["## Known limitations", ""]
    lines += ["{}. {}".format(i + 1, text) for i, text in enumerate(LIMITATIONS)]
    lines.append("")
    path = run_dir / "MODEL_REPORT.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_scoring_summary(out_dir: Path, info: dict, tier_counts: pd.DataFrame, district: pd.DataFrame) -> Path:
    lines = ["# Live Risk Scoring Summary", ""]
    lines += ["| Item | Value |", "|---|---|"]
    for key, value in info.items():
        lines.append("| {} | {} |".format(key, value))
    lines += ["", "## Students by tier", "", md_table(tier_counts)]
    lines += ["## Districts", "", md_table(district, floatfmt=".1f", max_rows=40)]
    lines += [
        "Tiers rank students who are attending (or whose status is not yet known): CRITICAL = top 2%, HIGH = next 5%, "
        "MEDIUM = next 10%. NOT_ATTENDING = the school reports attendance but the student has none this year. "
        "NO_SCHOOL_DATA = the school has not sent attendance. `expected_dropouts` is the sum of calibrated probabilities.",
        "",
    ]
    path = out_dir / "SCORING_SUMMARY.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
