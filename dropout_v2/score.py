"""Score the live academic year with the latest trained run.

    python -m dropout_v2.score                         # full data, snapshot chosen from the data that has arrived
    python -m dropout_v2.score --snapshot jul          # force a snapshot
    python -m dropout_v2.score --legacy-export         # also write artifacts/statewide_scores.parquet for the app
Refresh the live year first when new attendance arrives:
    python -m dropout_v2.extract --years 2026-27 --refresh
"""
from __future__ import annotations

import argparse
import datetime as _dt
import sys
import traceback
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from dropout_v2 import config, explain, features, modeling, report, utils
from dropout_v2.dataset import DataStore, rules_signature
from dropout_v2.tiers import assign_tiers  # noqa: F401  (re-exported for callers of score.assign_tiers)

KEY_FEATURES = ["cur_att_rate", "cur_trailing_zero_months", "prev_att_rate", "prev_status", "over_age",
                "school_prev_dropout_rate"]
LEGACY_LEVEL = {"CRITICAL": "High", "HIGH": "High", "MEDIUM": "Medium", "LOW": "Low", "NOT_ATTENDING": "High"}


def detect_available_months(base: pd.DataFrame, min_coverage: float = config.LIVE_MONTH_MIN_COVERAGE) -> int:
    """Number of leading months (from June) in which at least `min_coverage` of students have attendance."""
    if len(base) == 0:
        return 0
    count = 0
    for month in config.MONTHS:
        if base["wd_{}".format(month)].notna().mean() >= min_coverage:
            count += 1
        else:
            break
    return count


def choose_snapshot(available_months: int, trained: List[str]) -> str:
    fitting = [s for s in trained if config.SNAPSHOTS[s] <= available_months]
    if not fitting:
        raise ValueError("No trained snapshot fits {} available month(s). Trained: {}".format(available_months, trained))
    return max(fitting, key=lambda s: config.SNAPSHOTS[s])


def _reasons(bundle: modeling.ModelBundle, ds: pd.DataFrame, X: pd.DataFrame, chunk: int = 500_000) -> pd.DataFrame:
    parts = []
    values = ds[bundle.features]
    for start in range(0, len(X), chunk):
        contrib = explain.contributions(bundle.booster, X.iloc[start:start + chunk], bundle.best_iteration)
        parts.append(explain.top_reasons(contrib, values.iloc[start:start + chunk], k=config.REASONS_PER_STUDENT))
    return pd.concat(parts) if parts else pd.DataFrame(columns=["reason_{}".format(i + 1)
                                                               for i in range(config.REASONS_PER_STUDENT)])


def _legacy_export(scores: pd.DataFrame, path: Path) -> Path:
    tiered = scores[scores["risk_tier"].isin([name for name, _ in config.TIER_CUTS] + [config.TIER_REST])]
    high_cut = tiered.loc[tiered["risk_tier"].isin(["CRITICAL", "HIGH"]), "risk_probability"].min()
    medium_cut = tiered.loc[tiered["risk_tier"] == "MEDIUM", "risk_probability"].min()
    level = scores["risk_tier"].map(LEGACY_LEVEL)
    no_data = scores["risk_tier"] == "NO_SCHOOL_DATA"
    p = scores["risk_probability"]
    by_prob = np.where(p >= high_cut, "High", np.where(p >= medium_cut, "Medium", "Low"))
    level = level.where(~no_data, pd.Series(by_prob, index=scores.index)).fillna("Low")
    legacy = pd.DataFrame({
        "person_id": scores["person_id"].astype(str),
        "school_name": scores["school_name"].astype(str),
        "baseline_class": scores["class_t"].astype(int),
        "student_gender": scores["gender"].astype(str),
        "student_district_name": scores["district"].astype(str),
        "social_category": scores["social_category"].astype(str),
        "dwelling_type": scores["dwelling"].astype(str),
        "ml_risk_score": scores["risk_probability"].round(4),
        "ml_risk_level": level.astype(str),
        "risk_tier_v2": scores["risk_tier"].astype(str),
        "reason_1": scores["reason_1"], "reason_2": scores["reason_2"], "reason_3": scores["reason_3"],
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        path.replace(path.with_name("statewide_scores.backup_{}.parquet".format(stamp)))
    legacy.to_parquet(str(path), index=False)
    return path


def run_scoring(tag: str, run_dir: Optional[Path] = None, snapshot: str = "auto", live_year: str = config.LIVE_YEAR,
                legacy_export: bool = False, data_root: Optional[Path] = None, artifact_root: Optional[Path] = None,
                legacy_path: Optional[Path] = None) -> Path:
    run_dir = Path(run_dir) if run_dir else utils.read_latest_pointer(artifact_root)
    out_dir = run_dir / "scores"
    logger = utils.setup_logging(out_dir)
    trained = [s for s in config.SNAPSHOTS if (run_dir / "bundle_{}.joblib".format(s)).exists()]
    if not trained:
        raise FileNotFoundError("No model bundles in {}. Run: python -m dropout_v2.train".format(run_dir))
    store = DataStore(tag, root=data_root)
    live = store.clean(live_year)
    base = features.base_population(live)
    available = detect_available_months(base)
    if snapshot == "auto":
        snapshot = choose_snapshot(available, trained)
    elif snapshot not in trained:
        raise ValueError("Snapshot {} was not trained in {}. Trained: {}".format(snapshot, run_dir, trained))
    if config.SNAPSHOTS[snapshot] > available:
        logger.warning("Snapshot %s needs %d months but only %d months have attendance for >= %.0f%% of students.",
                       snapshot, config.SNAPSHOTS[snapshot], available, 100 * config.LIVE_MONTH_MIN_COVERAGE)
    logger.info("Live year %s: %d students in classes 1-9, %d month(s) of attendance available, snapshot %s",
                live_year, len(base), available, snapshot)
    bundle = modeling.load_bundle(run_dir / "bundle_{}.joblib".format(snapshot))
    if bundle.rules_signature and bundle.rules_signature != rules_signature():
        logger.warning("The model was trained with different data rules (%s) than the current code (%s). "
                       "Retrain for consistent results.", bundle.rules_signature, rules_signature())

    with utils.step("score {} {}".format(live_year, snapshot)):
        ds = store.dataset(live_year, snapshot, with_label=False)
        X = bundle.matrix(ds)
        probability = bundle.predict(X)
        reasons = _reasons(bundle, ds, X)
    school_names = store.schools()["school_name"]
    scores = pd.DataFrame({
        "person_id": ds["person_id"].astype(str),
        "student_id": ds["student_key"].astype("int64").astype(str).where(ds["student_key"] >= 0, ""),
        "academic_year": live_year,
        "snapshot": snapshot,
        "district": ds["district"].astype(str),
        "mandal": ds["mandal"].astype(str),
        "school_id": ds["school_key"].astype("int64").astype(str).where(ds["school_key"] >= 0, ""),
        "school_name": ds["school_key"].map(school_names).fillna("").astype(str),
        "class_t": ds["class_t"].astype(int),
        "gender": ds["gender"].astype(str),
        "social_category": ds["social_category"].astype(str),
        "dwelling": ds["dwelling"].astype(str),
        "mgmt_group": ds["mgmt_group"].astype(str),
        "engagement_status": ds["engagement_status"].astype(str),
        "risk_probability": probability.astype("float64"),
    })
    for col in KEY_FEATURES:
        if col in ds.columns:
            scores[col] = ds[col].astype(str) if isinstance(ds[col].dtype, pd.CategoricalDtype) else ds[col]
    scores = assign_tiers(scores, "risk_probability")
    scores = pd.concat([scores.reset_index(drop=True), reasons.reset_index(drop=True)], axis=1)

    out_dir.mkdir(parents=True, exist_ok=True)
    scores_path = out_dir / "scores_{}_{}.parquet".format(live_year, snapshot)
    scores.to_parquet(str(scores_path), index=False)

    tier_order = [name for name, _ in config.TIER_CUTS] + [config.TIER_REST, "NOT_ATTENDING", "NO_SCHOOL_DATA"]
    tier_counts = scores["risk_tier"].value_counts().reindex(tier_order).fillna(0).astype(int).reset_index()
    tier_counts.columns = ["tier", "students"]
    district = scores.pivot_table(index="district", columns="risk_tier", values="person_id", aggfunc="count",
                                  fill_value=0).reindex(columns=tier_order, fill_value=0)
    district.insert(0, "students", scores.groupby("district").size())
    district["expected_dropouts"] = scores.groupby("district")["risk_probability"].sum()
    district["mean_probability"] = scores.groupby("district")["risk_probability"].mean()
    district = district.sort_values("expected_dropouts", ascending=False).reset_index()
    district.to_csv(out_dir / "district_summary.csv", index=False)

    tiered = scores[scores["risk_tier"].isin([name for name, _ in config.TIER_CUTS] + [config.TIER_REST])]
    top = (tiered.sort_values("risk_probability", ascending=False)
           .groupby("district", sort=False).head(config.TOP_LIST_PER_DISTRICT)
           .sort_values(["district", "risk_probability"], ascending=[True, False]))
    list_cols = ["district", "mandal", "school_name", "school_id", "person_id", "student_id", "class_t", "gender",
                 "risk_probability", "risk_tier", "state_pct", "district_pct", "reason_1", "reason_2", "reason_3"]
    top[list_cols].to_csv(out_dir / "top_risk_by_district.csv", index=False)
    not_attending = scores[scores["risk_tier"] == "NOT_ATTENDING"].sort_values(
        ["district", "mandal", "school_name", "risk_probability"], ascending=[True, True, True, False])
    not_attending[list_cols].to_csv(out_dir / "not_attending.csv", index=False)

    info = {
        "run": run_dir.name, "live_year": live_year, "snapshot": snapshot,
        "months_available": available, "students_scored": int(len(scores)),
        "model_trained_on": ", ".join(bundle.train_years), "calibrated_on": bundle.calibration_year,
        "scored_at": _dt.datetime.now().isoformat(timespec="seconds"),
    }
    if legacy_export:
        path = _legacy_export(scores, legacy_path or config.LEGACY_SCORES_PATH)
        info["legacy_export"] = str(path)
        logger.info("Legacy scores for the Streamlit app: %s", path)
    utils.write_json(out_dir / "scoring_info.json", info)
    report.write_scoring_summary(out_dir, info, tier_counts, district)
    logger.info("Scores: %s", scores_path)
    return out_dir


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Score the live academic year with a trained run.")
    parser.add_argument("--sample-pct", type=int, default=0)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--tag", default=None)
    parser.add_argument("--run", default=None, help="run directory (default: artifacts/v2/LATEST)")
    parser.add_argument("--snapshot", default="auto", help="auto or one of " + ",".join(config.SNAPSHOTS))
    parser.add_argument("--live-year", default=config.LIVE_YEAR)
    parser.add_argument("--legacy-export", action="store_true",
                        help="write artifacts/statewide_scores.parquet for the existing Streamlit page")
    args = parser.parse_args(argv)
    tag = config.data_tag(args.sample_pct, args.synthetic, args.tag)
    try:
        out_dir = run_scoring(tag, Path(args.run) if args.run else None, args.snapshot, args.live_year,
                              args.legacy_export)
    except Exception as exc:
        utils.get_logger().error("Scoring failed: %s\n%s", exc, traceback.format_exc())
        print("SCORING FAILED: {}".format(exc), file=sys.stderr)
        return 1
    print("\nScoring finished. Summary: {}".format(out_dir / "SCORING_SUMMARY.md"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
