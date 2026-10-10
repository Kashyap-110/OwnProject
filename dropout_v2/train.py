"""Train one calibrated LightGBM model per snapshot and write the model report.

    python -m dropout_v2.train --sample-pct 2 --snapshots jul      # dry run on the 2% sample
    python -m dropout_v2.train --snapshots jul                      # full data, demo snapshot
    python -m dropout_v2.train                                      # all snapshots
    python -m dropout_v2.train --synthetic                          # no database needed
"""
from __future__ import annotations

import argparse
import datetime as _dt
import gc
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from dropout_v2 import config, evaluate, explain, modeling, report, synthetic, tiers, utils
from dropout_v2.dataset import DataStore, rules_signature
from dropout_v2.features import feature_columns

BREAKDOWNS = ["class_t", "district", "gender", "social_category", "mgmt_group", "engagement_status"]


def _versions() -> dict:
    import lightgbm
    import pyarrow
    import sklearn

    return {"python": sys.version.split()[0], "pandas": pd.__version__, "numpy": np.__version__,
            "scikit-learn": sklearn.__version__, "lightgbm": lightgbm.__version__, "pyarrow": pyarrow.__version__}


def _label_reports(store: DataStore) -> (pd.DataFrame, pd.DataFrame):
    detail, summary = [], []
    for year in store.years_available():
        lab = store.labels(year)
        if lab is None:
            continue
        counts = lab.groupby(["class_t", "label_reason"], observed=True).size().reset_index(name="students")
        counts.insert(0, "base_year", year)
        detail.append(counts)
        labeled = lab["label"].notna()
        src = lab["src_is_dropout"].astype(str)
        reason = lab["label_reason"].astype(str)
        summary.append({
            "base_year": year,
            "label_from": config.next_year(year),
            "base_students": int(len(lab)),
            "labeled": int(labeled.sum()),
            "dropout_rate": float(lab.loc[labeled, "label"].mean()) if labeled.any() else np.nan,
            "unknown_share": float(1 - labeled.mean()),
            "not_found_share": float((reason == "DROPOUT_NOT_FOUND").mean()),
            "not_attending_share": float((reason == "DROPOUT_NOT_ATTENDING").mean()),
            "class11_share": float((reason == "CONTINUED_CLASS_11_PLUS").mean()),
            "src_is_dropout_true_share": float((src == "TRUE").mean()),
            "dropout_rate_if_src_true": float(lab.loc[labeled & (src == "TRUE"), "label"].mean())
            if (labeled & (src == "TRUE")).any() else np.nan,
            "dropout_rate_if_src_false": float(lab.loc[labeled & (src == "FALSE"), "label"].mean())
            if (labeled & (src == "FALSE")).any() else np.nan,
        })
    detail_df = pd.concat(detail, ignore_index=True) if detail else pd.DataFrame()
    return detail_df, pd.DataFrame(summary)


def _labeled(ds: pd.DataFrame) -> pd.DataFrame:
    return ds.loc[ds["label"].notna()].reset_index(drop=True)


def _scaled_params(params: dict, n_rows: int) -> dict:
    out = dict(params)
    scaled = max(20, int(n_rows) // 200)
    out["min_data_in_leaf"] = min(int(out["min_data_in_leaf"]), scaled)
    out["min_data_per_group"] = min(int(out["min_data_per_group"]), scaled)
    return out


def _tier_table(test: pd.DataFrame, p: np.ndarray) -> pd.DataFrame:
    frame = test[["engagement_status", "district", "label"]].copy()
    frame["risk_probability"] = p
    tiered = tiers.assign_tiers(frame, "risk_probability")
    total_pos = max(1.0, float(tiered["label"].sum()))
    order = [name for name, _ in config.TIER_CUTS] + [config.TIER_REST, "NOT_ATTENDING", "NO_SCHOOL_DATA"]
    rows = []
    for name in order:
        part = tiered[tiered["risk_tier"] == name]
        if part.empty:
            continue
        rows.append({"tier": name, "students": int(len(part)), "observed_dropout_rate": float(part["label"].mean()),
                     "mean_predicted": float(part["risk_probability"].mean()),
                     "share_of_all_dropouts": float(part["label"].sum() / total_pos)})
    return pd.DataFrame(rows)


def train_snapshot(store: DataStore, snapshot: str, train_years: List[str], run_dir: Path, tune: int = 0,
                   refit: bool = True, bootstrap_reps: int = config.BOOTSTRAP_REPS, baselines: bool = True) -> dict:
    logger = utils.get_logger()
    numeric, categorical = feature_columns(snapshot)
    with utils.step("datasets for snapshot {}".format(snapshot)):
        train = pd.concat([_labeled(store.dataset(y, snapshot)) for y in train_years], ignore_index=True)
        valid = _labeled(store.dataset(config.VALID_YEAR, snapshot))
        test = _labeled(store.dataset(config.TEST_YEAR, snapshot))
    for name, frame in (("train", train), ("validation", valid), ("test", test)):
        positives = int(frame["label"].sum())
        logger.info("%s rows %d, positives %d (%.2f%%)", name, len(frame), positives,
                    100.0 * positives / max(1, len(frame)))
        if positives == 0:
            raise ValueError("No dropouts in the {} data for snapshot {}. Check the labels.".format(name, snapshot))
        if positives < 1000:
            logger.warning("Only %d positives in %s data: metrics will be unstable.", positives, name)

    min_count = max(config.RARE_CATEGORY_MIN_COUNT, int(config.RARE_CATEGORY_FRACTION * len(train)))
    categories = modeling.fit_categories(train, categorical, min_count)
    y_train = train["label"].to_numpy(dtype="float64")
    y_valid = valid["label"].to_numpy(dtype="float64")
    y_test = test["label"].to_numpy(dtype="float64")
    X_train = modeling.prepare_matrix(train, numeric, categorical, categories)
    X_valid = modeling.prepare_matrix(valid, numeric, categorical, categories)
    X_test = modeling.prepare_matrix(test, numeric, categorical, categories)

    mask, weights, rate = modeling.negative_sample(train["person_key"].to_numpy(), y_train, config.NEG_PER_POS,
                                                   config.NEG_RATE_MIN)
    Xs, ys, ws = X_train[mask], y_train[mask], weights[mask]
    params = _scaled_params(config.LGB_PARAMS, int(mask.sum()))

    rng = np.random.default_rng(config.RANDOM_SEED)
    if len(X_valid) > config.EARLY_STOP_VALID_ROWS:
        es_idx = np.sort(rng.choice(len(X_valid), size=config.EARLY_STOP_VALID_ROWS, replace=False))
    else:
        es_idx = np.arange(len(X_valid))
    X_es, y_es = X_valid.iloc[es_idx], y_valid[es_idx]

    if tune > 0:
        with utils.step("tuning {} trials".format(tune)):
            t_idx = rng.random(len(Xs)) < config.TUNE_SAMPLE_FRACTION
            params = modeling.tune_params(Xs[t_idx], ys[t_idx], ws[t_idx], X_es, y_es, params, tune)
    with utils.step("train evaluation model {}".format(snapshot)):
        booster, best = modeling.train_with_early_stopping(Xs, ys, ws, X_es, y_es, params, config.MAX_ROUNDS,
                                                           config.EARLY_STOP_PATIENCE)
    raw_valid = booster.predict(X_valid, num_iteration=best)
    calibrator = modeling.fit_calibrator(raw_valid, y_valid)
    eval_bundle = modeling.ModelBundle(
        snapshot=snapshot, numeric=numeric, categorical=categorical, categories=categories, booster=booster,
        best_iteration=best, calibrator=calibrator, neg_rate=rate, train_years=list(train_years),
        calibration_year=config.VALID_YEAR, params=params, rules_signature=rules_signature())
    modeling.save_bundle(eval_bundle, run_dir / "bundle_{}_eval.joblib".format(snapshot), write_text_model=False)

    with utils.step("evaluate {}".format(snapshot)):
        p_valid = calibrator.predict(raw_valid)
        p_test = eval_bundle.predict(X_test)
        metrics = {
            "snapshot": snapshot, "train_years": train_years, "valid_year": config.VALID_YEAR,
            "test_year": config.TEST_YEAR, "n_train": int(len(train)), "n_train_sampled": int(mask.sum()),
            "neg_rate": rate, "n_valid": int(len(valid)), "n_test": int(len(test)), "best_iteration": best,
            "params": {k: params[k] for k in ("learning_rate", "num_leaves", "min_data_in_leaf", "feature_fraction",
                                               "lambda_l2")},
            "valid": evaluate.core_metrics(y_valid, p_valid),
            "test": evaluate.core_metrics(y_test, p_test),
        }
        topk = evaluate.topk_table(y_test, p_test, config.TOP_K)
        topk.to_csv(run_dir / "topk_{}.csv".format(snapshot), index=False)
        metrics["topk"] = topk.to_dict(orient="records")
        topk_att = None
        attending = (test["engagement_status"].astype(str) == "ATTENDING").to_numpy()
        if config.visible_months(snapshot) and attending.any():
            metrics["attending"] = evaluate.core_metrics(y_test[attending], p_test[attending])
            topk_att = evaluate.topk_table(y_test[attending], p_test[attending], config.TOP_K)
            topk_att.to_csv(run_dir / "topk_{}_attending.csv".format(snapshot), index=False)
        calib = evaluate.calibration_table(y_test, p_test, bins=10)
        calib.to_csv(run_dir / "calibration_{}.csv".format(snapshot), index=False)
        tier_table = _tier_table(test, p_test)
        tier_table.to_csv(run_dir / "tiers_{}.csv".format(snapshot), index=False)
        metrics["tiers"] = tier_table.to_dict(orient="records")
        frame = test[BREAKDOWNS].copy()
        frame["label"] = y_test
        frame["p"] = p_test
        breakdowns = {}
        for by in BREAKDOWNS:
            table = evaluate.breakdown(frame, "label", "p", by, k=0.10)
            table.to_csv(run_dir / "breakdown_{}_{}.csv".format(snapshot, by), index=False)
            breakdowns[by] = table
        if bootstrap_reps > 0:
            metrics["bootstrap_ci"] = evaluate.bootstrap_ci(y_test, p_test, bootstrap_reps, config.BOOTSTRAP_MAX_ROWS,
                                                            seed=config.RANDOM_SEED)
        rule = evaluate.rule_baseline(test, snapshot)
        metrics["rule_baseline"] = evaluate.core_metrics(y_test, rule)
        curves = {"model": p_test, "attendance rule": rule}
        if baselines:
            logit = evaluate.logistic_baseline(train, y_train, test, numeric, categorical, config.LOGISTIC_SAMPLE_ROWS,
                                               seed=config.RANDOM_SEED)
            metrics["logistic_baseline"] = evaluate.core_metrics(y_test, logit)
            curves["logistic regression"] = logit
        single = evaluate.single_feature_auc(X_valid, y_valid)
        single.to_csv(run_dir / "single_feature_auc_{}.csv".format(snapshot), index=False)
        metrics["leakage_alerts"] = single.loc[single["auc"] > config.SINGLE_FEATURE_AUC_ALERT, "feature"].tolist()
        importance = explain.global_importance(booster, X_test, best)
        importance.to_csv(run_dir / "importance_{}.csv".format(snapshot), index=False)
        evaluate.plot_pr_curve(y_test, curves, run_dir / "pr_curve_{}.png".format(snapshot),
                               "Precision-recall, snapshot {} (test {})".format(snapshot, config.TEST_YEAR))
        evaluate.plot_calibration(calib, run_dir / "calibration_{}.png".format(snapshot),
                                  "Calibration, snapshot {}".format(snapshot))
        evaluate.plot_importance(importance, run_dir / "importance_{}.png".format(snapshot),
                                 "Top features, snapshot {}".format(snapshot))
    logger.info("snapshot %s: test AUC %.4f, AP %.4f, base rate %.4f", snapshot, metrics["test"]["auc"],
                metrics["test"]["ap"], metrics["test"]["positive_rate"])

    if refit:
        with utils.step("refit production model {}".format(snapshot)):
            # Concatenate the prepared matrices (same categories), not the wide feature frames: less memory.
            keys_prod = np.concatenate([train["person_key"].to_numpy(), valid["person_key"].to_numpy()])
            del train, valid, Xs
            gc.collect()
            X_prod = pd.concat([X_train, X_valid], ignore_index=True)
            y_prod = np.concatenate([y_train, y_valid])
            p_mask, p_weights, p_rate = modeling.negative_sample(keys_prod, y_prod, config.NEG_PER_POS,
                                                                 config.NEG_RATE_MIN)
            n_rounds = max(1, int(round(best * config.REFIT_ROUNDS_FACTOR)))
            prod_params = _scaled_params(params, int(p_mask.sum()))
            prod_booster = modeling.train_fixed(X_prod[p_mask], y_prod[p_mask], p_weights[p_mask], prod_params, n_rounds)
            prod_calibrator = modeling.fit_calibrator(prod_booster.predict(X_test, num_iteration=n_rounds), y_test)
            prod = modeling.ModelBundle(
                snapshot=snapshot, numeric=numeric, categorical=categorical, categories=categories,
                booster=prod_booster, best_iteration=n_rounds, calibrator=prod_calibrator, neg_rate=p_rate,
                train_years=list(train_years) + [config.VALID_YEAR], calibration_year=config.TEST_YEAR,
                params=prod_params, rules_signature=rules_signature())
            del X_prod
    else:
        prod = eval_bundle
        del train, valid, Xs
    metrics["refit"] = bool(refit)
    modeling.save_bundle(prod, run_dir / "bundle_{}.joblib".format(snapshot))
    utils.write_json(run_dir / "metrics_{}.json".format(snapshot), metrics)
    del X_train, X_valid, X_test, test
    gc.collect()
    return {"metrics": metrics, "topk": topk, "topk_attending": topk_att, "tiers": tier_table, "calibration": calib,
            "breakdowns": breakdowns, "importance": importance, "single_auc": single}


def run_training(tag: str, snapshots: Optional[List[str]] = None, tune: int = 0, refit: bool = True,
                 include_2022: bool = False, bootstrap_reps: int = config.BOOTSTRAP_REPS, baselines: bool = True,
                 run_dir: Optional[Path] = None, data_root: Optional[Path] = None,
                 artifact_root: Optional[Path] = None) -> Path:
    snapshots = list(snapshots or config.DEFAULT_SNAPSHOTS)
    unknown = [s for s in snapshots if s not in config.SNAPSHOTS]
    if unknown:
        raise ValueError("Unknown snapshots {}. Known: {}".format(unknown, list(config.SNAPSHOTS)))
    run_dir = run_dir or utils.new_run_dir(tag, artifact_root)
    logger = utils.setup_logging(run_dir)
    store = DataStore(tag, root=data_root)
    available = store.years_available()
    needed = config.TRAIN_YEARS + [config.VALID_YEAR, config.TEST_YEAR, config.next_year(config.TEST_YEAR)]
    missing = [y for y in needed if y not in available]
    if missing:
        raise FileNotFoundError("Data for {} is missing under {}. Run: python -m dropout_v2.extract".format(
            missing, store.base_dir))
    train_years = list(config.TRAIN_YEARS)
    if include_2022 and "2022-23" in available:
        train_years = ["2022-23"] + train_years
    context = {"tag": tag, "created": _dt.datetime.now().isoformat(timespec="seconds"),
               "rules_signature": rules_signature(), "train_years": train_years, "valid_year": config.VALID_YEAR,
               "test_year": config.TEST_YEAR}
    utils.write_json(run_dir / "config.json", dict(context, snapshots=snapshots, tune=tune, refit=refit,
                                                    bootstrap_reps=bootstrap_reps, baselines=baselines,
                                                    label_window=config.LABEL_WINDOW,
                                                    base_classes=list(config.BASE_CLASSES),
                                                    lgb_params=config.LGB_PARAMS, versions=_versions()))
    logger.info("Run directory: %s", run_dir)
    with utils.step("clean all years"):
        for year in available:
            store.clean(year)
    profile = store.profile()
    profile.to_csv(run_dir / "data_profile.csv", index=False)
    (run_dir / "data_profile.md").write_text("# Data profile\n\n" + utils.md_table(profile, floatfmt=".3f"),
                                             encoding="utf-8")
    with utils.step("labels"):
        label_detail, label_summary = _label_reports(store)
    label_detail.to_csv(run_dir / "label_report.csv", index=False)
    label_summary.to_csv(run_dir / "label_summary.csv", index=False)
    context.update(profile=profile, label_summary=label_summary)
    for row in label_summary.to_dict(orient="records"):
        if row["dropout_rate"] == row["dropout_rate"] and not (0.005 <= row["dropout_rate"] <= 0.30):
            logger.warning("Base year %s: dropout rate %.4f is outside 0.5%%-30%%. Check the label.",
                           row["base_year"], row["dropout_rate"])
        if row["unknown_share"] > 0.30:
            logger.warning("Base year %s: %.1f%% of labels are unknown (non-reporting schools).",
                           row["base_year"], 100 * row["unknown_share"])
    results: Dict[str, dict] = {}
    for snapshot in snapshots:
        with utils.step("snapshot {}".format(snapshot)):
            results[snapshot] = train_snapshot(store, snapshot, train_years, run_dir, tune=tune, refit=refit,
                                               bootstrap_reps=bootstrap_reps, baselines=baselines)
    report_path = report.write_model_report(run_dir, context, results)
    utils.write_latest_pointer(run_dir, artifact_root)
    logger.info("Model report: %s", report_path)
    return run_dir


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Train the v2 dropout models (one per snapshot).")
    parser.add_argument("--sample-pct", type=int, default=0, help="use the data extracted with this sample percent")
    parser.add_argument("--synthetic", action="store_true", help="use synthetic data (no database)")
    parser.add_argument("--synthetic-students", type=int, default=20000)
    parser.add_argument("--tag", default=None, help="explicit data tag (folder under data/v2)")
    parser.add_argument("--snapshots", default=",".join(config.DEFAULT_SNAPSHOTS),
                        help="comma list from: " + ",".join(config.SNAPSHOTS))
    parser.add_argument("--tune", type=int, default=0, help="random-search trials (0 = off)")
    parser.add_argument("--no-refit", action="store_true", help="score with the evaluation model")
    parser.add_argument("--include-2022", action="store_true", help="also train on base year 2022-23")
    parser.add_argument("--bootstrap", type=int, default=config.BOOTSTRAP_REPS)
    parser.add_argument("--no-baselines", action="store_true", help="skip the logistic-regression baseline")
    args = parser.parse_args(argv)
    tag = config.data_tag(args.sample_pct, args.synthetic, args.tag)
    try:
        if args.synthetic and not (utils.data_dir(tag) / "raw" / "schools.parquet").exists():
            print("Writing synthetic data ({} students)...".format(args.synthetic_students))
            synthetic.write_synthetic_cache(tag=tag, n_students=args.synthetic_students)
        run_dir = run_training(tag, [s.strip() for s in args.snapshots.split(",") if s.strip()], tune=args.tune,
                               refit=not args.no_refit, include_2022=args.include_2022,
                               bootstrap_reps=args.bootstrap, baselines=not args.no_baselines)
    except Exception as exc:  # report clearly and fail the command
        utils.get_logger().error("Training failed: %s\n%s", exc, traceback.format_exc())
        print("TRAINING FAILED: {}".format(exc), file=sys.stderr)
        return 1
    print("\nTraining finished. Report: {}".format(run_dir / "MODEL_REPORT.md"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
