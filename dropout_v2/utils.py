"""Small helpers shared by all modules: logging, timing, paths, Markdown and JSON output."""
from __future__ import annotations

import contextlib
import datetime as _dt
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import pandas as pd

from dropout_v2 import config

LOGGER_NAME = "dropout_v2"


def get_logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME)


def setup_logging(run_dir: Optional[Path] = None, level: int = logging.INFO) -> logging.Logger:
    """Log to the console and, when run_dir is given, to run_dir/run.log."""
    logger = get_logger()
    logger.setLevel(level)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%Y-%m-%d %H:%M:%S")
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(console)
    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(run_dir / "run.log"), encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
    logger.propagate = False
    return logger


def peak_memory_gb() -> float:
    """Peak resident memory of this process in GB (Linux/macOS); NaN where unavailable."""
    try:
        import resource  # not available on Windows

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak / (1024.0 ** 2) if sys.platform != "darwin" else peak / (1024.0 ** 3)
    except Exception:
        return float("nan")


@contextlib.contextmanager
def step(name: str) -> Iterator[None]:
    """Log the start, duration and peak memory of a pipeline step."""
    logger = get_logger()
    logger.info("START %s", name)
    started = time.time()
    try:
        yield
    finally:
        mem = peak_memory_gb()
        mem_txt = "" if math.isnan(mem) else ", peak memory {:.1f} GB".format(mem)
        logger.info("DONE  %s in %.1f s%s", name, time.time() - started, mem_txt)


def data_dir(tag: str, root: Optional[Path] = None) -> Path:
    return (root or config.DATA_ROOT) / tag


def new_run_dir(tag: str, root: Optional[Path] = None) -> Path:
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = (root or config.ARTIFACT_ROOT) / "{}_{}".format(stamp, tag)
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_latest_pointer(run_dir: Path, root: Optional[Path] = None) -> None:
    base = root or config.ARTIFACT_ROOT
    base.mkdir(parents=True, exist_ok=True)
    (base / "LATEST").write_text(str(run_dir.resolve()), encoding="utf-8")


def read_latest_pointer(root: Optional[Path] = None) -> Path:
    pointer = (root or config.ARTIFACT_ROOT) / "LATEST"
    if not pointer.exists():
        raise FileNotFoundError("No trained run found ({} is missing). Run: python -m dropout_v2.train".format(pointer))
    return Path(pointer.read_text(encoding="utf-8").strip())


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return None if math.isnan(value) else value
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (Path,)):
        return str(obj)
    if isinstance(obj, (pd.Timestamp, _dt.date, _dt.datetime)):
        return obj.isoformat()
    return str(obj)


def _clean_nan(obj):
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if isinstance(obj, dict):
        return {str(k): _clean_nan(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean_nan(v) for v in obj]
    return obj


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean_nan(obj), indent=2, default=_json_default), encoding="utf-8")


def md_table(df: pd.DataFrame, floatfmt: str = ".4f", max_rows: int = 60) -> str:
    """Render a DataFrame as a GitHub Markdown table (no tabulate dependency)."""
    if df is None or len(df) == 0:
        return "_(no rows)_\n"
    view = df.head(max_rows)

    def fmt(value) -> str:
        if value is None:
            return ""
        if isinstance(value, (float, np.floating)):
            if math.isnan(float(value)):
                return ""
            return format(float(value), floatfmt)
        if isinstance(value, (int, np.integer)):
            return "{:,}".format(int(value))
        return str(value).replace("|", "/")

    header = "| " + " | ".join(str(c) for c in view.columns) + " |"
    sep = "|" + "|".join("---" for _ in view.columns) + "|"
    lines = [header, sep]
    for row in view.itertuples(index=False):
        lines.append("| " + " | ".join(fmt(v) for v in row) + " |")
    if len(df) > max_rows:
        lines.append("\n_({} more rows not shown)_".format(len(df) - max_rows))
    return "\n".join(lines) + "\n"
