"""LightGBM training, calibration and the saved model bundle."""
from __future__ import annotations

import dataclasses
import datetime as _dt
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score

from dropout_v2 import config, utils


# --------------------------------------------------------------------------- matrices
def fit_categories(df: pd.DataFrame, categorical: List[str], min_count: int) -> Dict[str, List[str]]:
    """Allowed categories per column: values seen at least `min_count` times, then OTHER, UNKNOWN."""
    out: Dict[str, List[str]] = {}
    for col in categorical:
        values = df[col].astype(object).where(df[col].notna(), config.UNKNOWN).astype(str)
        counts = values.value_counts()
        keep = [v for v in counts.index if counts[v] >= min_count and v not in (config.OTHER, config.UNKNOWN)]
        out[col] = list(keep) + [config.OTHER, config.UNKNOWN]
    return out


def _to_fixed_categories(values: pd.Series, allowed: List[str]) -> pd.Categorical:
    allowed_set = set(allowed)
    if isinstance(values.dtype, pd.CategoricalDtype):
        cats = [str(c) for c in values.cat.categories]
        mapped = np.array([c if c in allowed_set else config.OTHER for c in cats] + [config.UNKNOWN], dtype=object)
        mapped = np.where(mapped == "nan", config.UNKNOWN, mapped)
        out = mapped[values.cat.codes.to_numpy()]
    else:
        text = values.astype(object).where(values.notna(), config.UNKNOWN).astype(str)
        out = np.where(text.isin(allowed_set), text, config.OTHER)
    return pd.Categorical(out, categories=allowed)


def prepare_matrix(df: pd.DataFrame, numeric: List[str], categorical: List[str],
                   categories: Dict[str, List[str]]) -> pd.DataFrame:
    """Model input: float32 numerics and categoricals with the training category list."""
    missing = [c for c in numeric + categorical if c not in df.columns]
    if missing:
        raise KeyError("Feature columns missing from the dataset: {}".format(missing))
    data = {}
    for col in numeric:
        data[col] = pd.to_numeric(df[col], errors="coerce").astype("float32").to_numpy()
    for col in categorical:
        data[col] = _to_fixed_categories(df[col], categories[col])
    return pd.DataFrame(data, index=df.index, columns=numeric + categorical)


# --------------------------------------------------------------------------- sampling
def negative_sample(keys: np.ndarray, y: np.ndarray, neg_per_pos: float, rate_min: float,
                    seed: int = config.RANDOM_SEED) -> Tuple[np.ndarray, np.ndarray, float]:
    """Keep all positives and a deterministic share of negatives. Returns (mask, weights, rate).

    weights is aligned to the full input: 1 for positives, 1/rate for negatives.
    """
    y = np.asarray(y)
    pos = y == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    rate = 1.0 if n_neg == 0 else float(min(1.0, max(rate_min, neg_per_pos * n_pos / n_neg)))
    hashed = (np.asarray(keys).astype("uint64") * np.uint64(11400714819323198485) + np.uint64(seed))
    u = (hashed >> np.uint64(11)).astype("float64") / float(2 ** 53)
    mask = pos | (u < rate)
    weights = np.where(pos, 1.0, 1.0 / rate)
    return mask, weights, rate


# --------------------------------------------------------------------------- training
def _log_every(period: int):
    logger = utils.get_logger()

    def _callback(env):
        if period and (env.iteration + 1) % period == 0 and env.evaluation_result_list:
            parts = ["{} {}={:.5f}".format(name, metric, value) for name, metric, value, _ in env.evaluation_result_list]
            logger.info("  round %d: %s", env.iteration + 1, ", ".join(parts))

    return _callback


def _categorical_names(X: pd.DataFrame) -> List[str]:
    return [c for c in X.columns if isinstance(X[c].dtype, pd.CategoricalDtype)]


def train_with_early_stopping(Xtr: pd.DataFrame, ytr: np.ndarray, wtr: np.ndarray, Xva: pd.DataFrame,
                              yva: np.ndarray, params: dict, max_rounds: int, patience: int
                              ) -> Tuple[lgb.Booster, int]:
    cats = _categorical_names(Xtr)
    dtrain = lgb.Dataset(Xtr, label=ytr, weight=wtr, categorical_feature=cats, free_raw_data=True)
    dvalid = lgb.Dataset(Xva, label=yva, reference=dtrain, categorical_feature=cats, free_raw_data=True)
    booster = lgb.train(
        params, dtrain, num_boost_round=max_rounds, valid_sets=[dvalid], valid_names=["valid"],
        callbacks=[lgb.early_stopping(patience, first_metric_only=True, verbose=False), _log_every(100)])
    best = int(booster.best_iteration or booster.current_iteration())
    return booster, max(1, best)


def train_fixed(X: pd.DataFrame, y: np.ndarray, w: np.ndarray, params: dict, n_rounds: int) -> lgb.Booster:
    cats = _categorical_names(X)
    dtrain = lgb.Dataset(X, label=y, weight=w, categorical_feature=cats, free_raw_data=True)
    return lgb.train(params, dtrain, num_boost_round=int(n_rounds), callbacks=[_log_every(100)])


def fit_calibrator(raw: np.ndarray, y: np.ndarray) -> IsotonicRegression:
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calibrator.fit(np.asarray(raw, dtype="float64"), np.asarray(y, dtype="float64"))
    return calibrator


def tune_params(Xtr, ytr, wtr, Xva, yva, base_params: dict, n_trials: int, seed: int = config.RANDOM_SEED) -> dict:
    """Small random search on validation average precision. Returns the best parameter set."""
    rng = np.random.default_rng(seed)
    logger = utils.get_logger()
    best_params, best_score = dict(base_params), -1.0
    for trial in range(n_trials):
        params = dict(base_params)
        for key, options in config.TUNE_GRID.items():
            params[key] = options[int(rng.integers(len(options)))]
        booster, best_iter = train_with_early_stopping(Xtr, ytr, wtr, Xva, yva, params, 1500, 50)
        score = average_precision_score(yva, booster.predict(Xva, num_iteration=best_iter))
        logger.info("tune trial %d/%d AP=%.5f params=%s", trial + 1, n_trials, score,
                    {k: params[k] for k in config.TUNE_GRID})
        if score > best_score:
            best_score, best_params = score, params
    return best_params


# --------------------------------------------------------------------------- bundle
@dataclasses.dataclass
class ModelBundle:
    snapshot: str
    numeric: List[str]
    categorical: List[str]
    categories: Dict[str, List[str]]
    booster: lgb.Booster
    best_iteration: int
    calibrator: IsotonicRegression
    neg_rate: float
    train_years: List[str]
    calibration_year: Optional[str]
    params: dict
    created: str = dataclasses.field(default_factory=lambda: _dt.datetime.now().isoformat(timespec="seconds"))
    rules_signature: str = ""

    @property
    def features(self) -> List[str]:
        return list(self.numeric) + list(self.categorical)

    def matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        return prepare_matrix(df, self.numeric, self.categorical, self.categories)

    def predict_raw(self, df: pd.DataFrame) -> np.ndarray:
        X = df if list(df.columns) == self.features and _is_matrix(df) else self.matrix(df)
        return self.booster.predict(X, num_iteration=self.best_iteration)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.calibrator.predict(self.predict_raw(df))


def _is_matrix(df: pd.DataFrame) -> bool:
    return all(df[c].dtype == np.float32 or isinstance(df[c].dtype, pd.CategoricalDtype) for c in df.columns)


def save_bundle(bundle: ModelBundle, path: Path, write_text_model: bool = True) -> None:
    """Pickle the bundle; also write the LightGBM text model (model_<snapshot>.txt) for inspection."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, str(path), compress=3)
    if write_text_model:
        bundle.booster.save_model(str(path.parent / "model_{}.txt".format(bundle.snapshot)),
                                  num_iteration=bundle.best_iteration)


def load_bundle(path: Path) -> ModelBundle:
    bundle = joblib.load(str(path))
    if not isinstance(bundle, ModelBundle):
        raise TypeError("{} is not a dropout_v2 model bundle".format(path))
    return bundle
