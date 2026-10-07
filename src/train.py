import logging
import json
from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
import joblib
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns


logger = logging.getLogger("pipeline")


# ---------------------------------------------------------
# 1. Pipeline Factory
# ---------------------------------------------------------

def build_pipeline(model_name, config, numeric_cols, categorical_cols):
    """
    Build a full sklearn Pipeline: preprocessing + model.
    """

    # Numeric preprocessing
    num_prep = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler())
    ])

    # Categorical preprocessing
    cat_prep = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    prep = ColumnTransformer([
        ("num", num_prep, numeric_cols),
        ("cat", cat_prep, categorical_cols),
    ])

    # Models
    models = {
        "ridge": Ridge(alpha=config["model"]["ridge_alpha"]),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            learning_rate=config["model"]["hgb_learning_rate"],
            max_depth=config["model"]["hgb_max_depth"],
            random_state=42
        )
    }

    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}")

    model = models[model_name]

    pipe = Pipeline([
        ("prep", prep),
        ("model", model)
    ])

    return pipe


# ---------------------------------------------------------
# 2. Time-aware CV
# ---------------------------------------------------------

def run_time_aware_cv(pipe, X_train, y_train):
    """
    Expanding-window CV using TimeSeriesSplit.
    """
    tscv = TimeSeriesSplit(n_splits=3)
    scores = []

    for train_idx, val_idx in tscv.split(X_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_val)
        pred = np.clip(pred, 0, None)
        mae = mean_absolute_error(y_val, pred)
        scores.append(mae)

    return np.mean(scores), np.std(scores)


# ---------------------------------------------------------
# 3. Train + Evaluate + Save
# ---------------------------------------------------------

def run_train(config):
    processed = Path(config["paths"]["processed"])
    results = Path(config["paths"]["results"])
    models_dir = Path(config["paths"]["models"])
    models_dir.mkdir(exist_ok=True)
    results.mkdir(exist_ok=True)

    # Load train and validation
    train = pd.read_parquet(processed / "train.parquet")
    val = pd.read_parquet(processed / "val.parquet")

    # Sort by time
    train = train.sort_values(["pickup_hour", "PULocationID"])
    val = val.sort_values(["pickup_hour", "PULocationID"])

    # Features
    target = "demand"
    feature_cols = [
        "hour", "day_of_week", "month", "is_weekend",
        "temperature_2m", "precipitation", "snowfall", "wind_speed_10m",
        "lag_1", "lag_24", "lag_168", "rolling_24",
        "Borough", "PULocationID"
    ]

    numeric_cols = [
        "hour", "day_of_week", "month", "is_weekend",
        "temperature_2m", "precipitation", "snowfall", "wind_speed_10m",
        "lag_1", "lag_24", "lag_168", "rolling_24"
    ]

    categorical_cols = ["Borough", "PULocationID"]

    X_train = train[feature_cols]
    y_train = train[target]

    X_val = val[feature_cols]
    y_val = val[target]

    baseline_mae = config["baseline_mae"]

    # Compare two models
    rows = []

    for model_name in ["ridge", "hist_gradient_boosting"]:
        pipe = build_pipeline(model_name, config, numeric_cols, categorical_cols)

        # CV
        cv_mean, cv_std = run_time_aware_cv(pipe, X_train, y_train)

        # Fit on full train
        pipe.fit(X_train, y_train)

        # Validation
        pred_val = pipe.predict(X_val)
        pred_val = np.clip(pred_val, 0, None)
        val_mae = mean_absolute_error(y_val, pred_val)

        # Save comparison row
        row = pd.DataFrame([{
            "date": pd.Timestamp.today().date(),
            "model": model_name,
            "cv_mae_mean": cv_mean,
            "cv_mae_std": cv_std,
            "val_mae": val_mae,
            "baseline_mae": baseline_mae
        }])

        path = results / "model_comparison.csv"
        row.to_csv(path, mode="a", header=not path.exists(), index=False)

        rows.append((model_name, val_mae, pipe))

    # Choose winner
    winner_name, winner_mae, winner_pipe = sorted(rows, key=lambda x: x[1])[0]

    # Error breakdown
    pred_val = winner_pipe.predict(X_val)
    pred_val = np.clip(pred_val, 0, None)
    val["pred"] = pred_val
    val["error"] = np.abs(val["demand"] - val["pred"])

    # ========================================
    # MAE by hour
    # ========================================
    hour_mae = val.groupby("hour")["error"].mean()
    hour_mae.to_csv(results / "mae_by_hour.csv")

    plt.figure(figsize=(10, 5))
    sns.lineplot(x=hour_mae.index, y=hour_mae.values, marker="o")
    plt.title("MAE by Hour of Day")
    plt.xlabel("Hour")
    plt.ylabel("MAE")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(results / "mae_by_hour.png", dpi=100)
    plt.close()

    # ========================================
    # MAE by borough
    # ========================================
    borough_mae = val.groupby("Borough")["error"].mean()
    borough_mae.to_csv(results / "mae_by_borough.csv")

    plt.figure(figsize=(10, 5))
    colors_borough = sns.color_palette("viridis", len(borough_mae))
    plt.bar(range(len(borough_mae)), borough_mae.values, color=colors_borough)
    plt.xticks(range(len(borough_mae)), borough_mae.index, rotation=45)
    plt.title("MAE by Borough")
    plt.xlabel("Borough")
    plt.ylabel("MAE")
    plt.tight_layout()
    plt.savefig(results / "mae_by_borough.png", dpi=100)
    plt.close()

    # ========================================
    # MAE by rain
    # ========================================
    rain_mae = val.groupby(val["precipitation"] > 0)["error"].mean()
    rain_labels = ["Dry", "Rain"]

    plt.figure(figsize=(6, 5))
    colors_rain = sns.color_palette("coolwarm", len(rain_mae))
    plt.bar(range(len(rain_mae)), rain_mae.values, color=colors_rain)
    plt.xticks(range(len(rain_mae)), rain_labels)
    plt.title("MAE: Rain vs Dry")
    plt.xlabel("Weather Condition")
    plt.ylabel("MAE")
    plt.tight_layout()
    plt.savefig(results / "mae_by_rain.png", dpi=100)
    plt.close()

    # ========================================
    # Save model
    # ========================================
    joblib.dump(winner_pipe, models_dir / "demand_model.joblib")

    metadata = {
        "model": winner_name,
        "train_start": "2026-01-01",
        "train_end": "2026-04-30",
        "features": feature_cols,
        "val_mae": float(winner_mae),
        "baseline_mae": float(baseline_mae),
        "sklearn_version": sklearn.__version__,
        "pandas_version": pd.__version__,
        "date": str(pd.Timestamp.today().date())
    }

    with open(models_dir / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Winner: {winner_name} (MAE={winner_mae:.4f})")
    logger.info("Model saved.")