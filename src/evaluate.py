"""
Final test evaluation (Week 5, Day 4).

The locked test month (June) is opened ONCE. Nothing is tuned afterwards.

Reports:
  - test MAE of the saved final model
  - test MAE of the baseline (same zone, same hour last week = lag_168)
  - validation numbers for comparison, and the val -> test gap

Output: results/test_results.csv
"""

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

logger = logging.getLogger("pipeline")


def _baseline_mae(df, name):
    """Baseline = demand one week earlier for the same zone and hour (lag_168)."""
    if "lag_168" not in df.columns:
        raise KeyError(f"Column 'lag_168' missing in {name} data: re-run the features stage")
    mask = df["lag_168"].notna()
    dropped = int((~mask).sum())
    if dropped:
        logger.warning(f"{name}: {dropped} rows without lag_168 excluded from baseline MAE")
    return mean_absolute_error(df.loc[mask, "demand"], df.loc[mask, "lag_168"])


def run_test(config):
    processed = Path(config["paths"]["processed"])
    results = Path(config["paths"]["results"])
    models_dir = Path(config["paths"]["models"])
    results.mkdir(exist_ok=True)

    out_path = results / "test_results.csv"

    # Guard: the test set may be evaluated only once
    if out_path.exists():
        raise RuntimeError(
            f"{out_path} already exists: the test set was already evaluated. "
            "Evaluating it again and acting on the result makes it an unfair test. "
            "Delete the file only if you are deliberately starting from scratch."
        )

    model_path = models_dir / "demand_model.joblib"
    meta_path = models_dir / "model_metadata.json"
    test_path = processed / "test.parquet"
    val_path = processed / "val.parquet"

    for p in (model_path, meta_path, test_path, val_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing {p}: run the earlier stages first (train, features)")

    model = joblib.load(model_path)
    with open(meta_path) as f:
        meta = json.load(f)
    feature_cols = meta["features"]

    test = pd.read_parquet(test_path)
    val = pd.read_parquet(val_path)
    logger.info(f"Final model: {meta['model']}")
    logger.info(f"Test rows: {len(test):,}")

    # Model on test
    pred = np.clip(model.predict(test[feature_cols]), 0, None)
    test_mae = mean_absolute_error(test["demand"], pred)

    # Baselines
    test_baseline = _baseline_mae(test, "test")
    val_baseline = _baseline_mae(val, "val")
    val_mae = float(meta["val_mae"])

    gap = test_mae - val_mae
    gain = (test_baseline - test_mae) / test_baseline * 100

    logger.info("=" * 60)
    logger.info("FINAL TEST RESULTS")
    logger.info(f"Model    val MAE: {val_mae:.4f} | test MAE: {test_mae:.4f}")
    logger.info(f"Baseline val MAE: {val_baseline:.4f} | test MAE: {test_baseline:.4f}")
    logger.info(f"Gap (test - val): {gap:+.4f}")
    logger.info(f"Model vs baseline on test: {gain:.1f}% lower MAE")
    logger.info("=" * 60)

    pd.DataFrame([
        {"model": meta["model"], "split": "validation", "mae": val_mae},
        {"model": meta["model"], "split": "test", "mae": test_mae},
        {"model": "baseline_lag_168", "split": "validation", "mae": val_baseline},
        {"model": "baseline_lag_168", "split": "test", "mae": test_baseline},
    ]).to_csv(out_path, index=False)

    logger.info(f"Saved {out_path}")
    logger.info("Test evaluated once. Do not change the model or features after this point.")