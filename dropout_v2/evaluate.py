"""Evaluation: metrics, top-k tables, calibration, group breakdowns, bootstrap CIs, baselines, plots."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve, roc_auc_score  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402

from dropout_v2 import config  # noqa: E402


def _safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    if len(y) == 0 or np.unique(y).size < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def _safe_ap(y: np.ndarray, p: np.ndarray) -> float:
    if len(y) == 0 or y.sum() == 0:
        return float("nan")
    return float(average_precision_score(y, p))


def core_metrics(y, p) -> Dict[str, float]:
    y = np.asarray(y, dtype="float64")
    p = np.asarray(p, dtype="float64")
    return {
        "n": int(len(y)),
        "positives": int(y.sum()),
        "positive_rate": float(y.mean()) if len(y) else float("nan"),
        "auc": _safe_auc(y, p),
        "ap": _safe_ap(y, p),
        "brier": float(brier_score_loss(y, np.clip(p, 0, 1))) if len(y) else float("nan"),
    }


def _top_order(p: np.ndarray) -> np.ndarray:
    return np.argsort(-p, kind="mergesort")


def topk_table(y, p, ks: List[float]) -> pd.DataFrame:
    """Precision, recall and lift when the top k share of students (by score) is flagged."""
    y = np.asarray(y, dtype="float64")
    p = np.asarray(p, dtype="float64")
    order = _top_order(p)
    positives = y.sum()
    base_rate = y.mean() if len(y) else float("nan")
    rows = []
    for k in ks:
        n_flag = max(1, int(math.ceil(k * len(y))))
        top = order[:n_flag]
        precision = float(y[top].mean())
        recall = float(y[top].sum() / positives) if positives > 0 else float("nan")
        rows.append({
            "top_k": k, "n_flagged": n_flag, "score_threshold": float(p[top[-1]]),
            "true_dropouts_flagged": int(y[top].sum()), "precision": precision, "recall": recall,
            "lift": precision / base_rate if base_rate > 0 else float("nan"),
        })
    return pd.DataFrame(rows)


def calibration_table(y, p, bins: int = 10) -> pd.DataFrame:
    y = np.asarray(y, dtype="float64")
    p = np.asarray(p, dtype="float64")
    frame = pd.DataFrame({"y": y, "p": p})
    frame["bin"] = pd.qcut(frame["p"].rank(method="first"), q=bins, labels=False)
    table = frame.groupby("bin").agg(n=("y", "size"), mean_predicted=("p", "mean"), observed_rate=("y", "mean"))
    return table.reset_index()


def breakdown(df: pd.DataFrame, y_col: str, p_col: str, by: str, k: float = 0.10) -> pd.DataFrame:
    """Per-group metrics; recall uses one statewide threshold (top k share of all rows)."""
    y_all = df[y_col].to_numpy(dtype="float64")
    p_all = df[p_col].to_numpy(dtype="float64")
    n_flag = max(1, int(math.ceil(k * len(df))))
    threshold = np.sort(p_all)[::-1][n_flag - 1] if len(df) else np.inf
    rows = []
    for group, part in df.groupby(by, observed=True, sort=True):
        y = part[y_col].to_numpy(dtype="float64")
        p = part[p_col].to_numpy(dtype="float64")
        flagged = p >= threshold
        rows.append({
            by: group, "n": int(len(y)), "positives": int(y.sum()),
            "positive_rate": float(y.mean()) if len(y) else float("nan"),
            "auc": _safe_auc(y, p), "ap": _safe_ap(y, p),
            "flagged_share": float(flagged.mean()) if len(y) else float("nan"),
            "recall_at_top_k": float(y[flagged].sum() / y.sum()) if y.sum() > 0 else float("nan"),
            "mean_predicted": float(p.mean()) if len(p) else float("nan"),
        })
    return pd.DataFrame(rows)


def bootstrap_ci(y, p, reps: int, max_rows: int, seed: int = 0, k: float = 0.05) -> Dict[str, tuple]:
    """95% bootstrap intervals for AUC, AP and recall at the top k share."""
    y = np.asarray(y, dtype="float64")
    p = np.asarray(p, dtype="float64")
    rng = np.random.default_rng(seed)
    if len(y) > max_rows:
        keep = rng.choice(len(y), size=max_rows, replace=False)
        y, p = y[keep], p[keep]
    stats = {"auc": [], "ap": [], "recall_top_{}".format(int(k * 100)): []}
    n = len(y)
    n_flag = max(1, int(math.ceil(k * n)))
    for _ in range(int(reps)):
        idx = rng.integers(0, n, n)
        ys, ps = y[idx], p[idx]
        stats["auc"].append(_safe_auc(ys, ps))
        stats["ap"].append(_safe_ap(ys, ps))
        top = _top_order(ps)[:n_flag]
        stats["recall_top_{}".format(int(k * 100))].append(ys[top].sum() / ys.sum() if ys.sum() > 0 else np.nan)
    return {key: (float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5)))
            for key, vals in stats.items() if len(vals)}


def single_feature_auc(X: pd.DataFrame, y, max_rows: int = 500_000, seed: int = 0) -> pd.DataFrame:
    """AUC of each numeric feature alone (max of AUC and 1-AUC). Very high values signal leakage."""
    y = np.asarray(y, dtype="float64")
    idx = np.arange(len(y))
    if len(y) > max_rows:
        idx = np.random.default_rng(seed).choice(len(y), size=max_rows, replace=False)
    rows = []
    for col in X.columns:
        if isinstance(X[col].dtype, pd.CategoricalDtype):
            continue
        values = X[col].to_numpy(dtype="float64")[idx]
        if np.isnan(values).all():
            continue
        values = np.where(np.isnan(values), np.nanmedian(values), values)
        auc = _safe_auc(y[idx], values)
        rows.append({"feature": col, "auc": max(auc, 1 - auc) if not math.isnan(auc) else auc})
    return pd.DataFrame(rows).sort_values("auc", ascending=False).reset_index(drop=True)


def rule_baseline(df: pd.DataFrame, snapshot: str) -> np.ndarray:
    """Simple rule officials could use without a model: low attendance first."""
    if config.visible_months(snapshot):
        return 1.0 - df["cur_att_rate"].fillna(0.0).to_numpy(dtype="float64")
    return 1.0 - df["prev_att_rate"].fillna(0.5).to_numpy(dtype="float64")


def logistic_baseline(train_df: pd.DataFrame, y_train, test_df: pd.DataFrame, numeric: List[str],
                      categorical: List[str], sample_rows: int, seed: int = 0) -> np.ndarray:
    y_train = np.asarray(y_train, dtype="float64")
    rng = np.random.default_rng(seed)
    idx = np.arange(len(train_df))
    if len(idx) > sample_rows:
        idx = rng.choice(len(idx), size=sample_rows, replace=False)

    def frame(df):
        out = df[numeric].astype("float64").copy()
        for col in categorical:
            out[col] = df[col].astype(object).where(df[col].notna(), config.UNKNOWN).astype(str)
        return out

    pre = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.001), categorical),
    ])
    model = Pipeline([("pre", pre), ("lr", LogisticRegression(max_iter=500, C=1.0))])
    all_nan = [c for c in numeric if train_df[c].iloc[idx].isna().all()]
    if all_nan:
        numeric = [c for c in numeric if c not in all_nan]
        pre.transformers[0] = ("num", pre.transformers[0][1], numeric)
    model.fit(frame(train_df.iloc[idx]), y_train[idx])
    scores = []
    chunk = 500_000
    for start in range(0, len(test_df), chunk):             # convert the test year piece by piece (memory)
        scores.append(model.predict_proba(frame(test_df.iloc[start:start + chunk]))[:, 1])
    return np.concatenate(scores) if scores else np.array([])


# --------------------------------------------------------------------------- plots
def plot_pr_curve(y, curves: Dict[str, np.ndarray], path: Path, title: str) -> Path:
    y = np.asarray(y, dtype="float64")
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, scores in curves.items():
        if y.sum() == 0:
            continue
        precision, recall, _ = precision_recall_curve(y, scores)
        ax.plot(recall, precision, label="{} (AP {:.3f})".format(name, _safe_ap(y, scores)))
    ax.axhline(y.mean() if len(y) else 0, color="grey", linestyle="--", linewidth=1, label="base rate")
    ax.set_xlabel("Recall (share of real dropouts found)")
    ax.set_ylabel("Precision (share of flagged students who drop out)")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(path), dpi=120)
    plt.close(fig)
    return path


def plot_calibration(table: pd.DataFrame, path: Path, title: str) -> Path:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    top = max(float(table["mean_predicted"].max()), float(table["observed_rate"].max()), 1e-3)
    ax.plot([0, top], [0, top], color="grey", linestyle="--", linewidth=1, label="perfect")
    ax.plot(table["mean_predicted"], table["observed_rate"], marker="o", label="model")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed dropout rate")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(path), dpi=120)
    plt.close(fig)
    return path


def plot_importance(importance: pd.DataFrame, path: Path, title: str, top: int = 25) -> Path:
    view = importance.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, max(3, 0.3 * len(view) + 1)))
    ax.barh(view["feature"], view["mean_abs_contribution"])
    ax.set_xlabel("Mean |contribution| (log-odds)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(str(path), dpi=120)
    plt.close(fig)
    return path
