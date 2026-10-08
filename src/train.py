import logging
import json
import os
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
        "ridge": Ridge(alpha=config["model"]["ridge_alpha"], random_state=42),
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

        path = os.fspath(results / "model_comparison.csv")
        row.to_csv(path, mode="a", header=not Path(path).exists(), index=False)

        rows.append((model_name, val_mae, pipe))

    # Choose winner
    winner_name, winner_mae, winner_pipe = sorted(rows, key=lambda x: x[1])[0]

    # Error breakdown
    pred_val = winner_pipe.predict(X_val)
    pred_val = np.clip(pred_val, 0, None)
    val["pred"] = pred_val
    val["error"] = np.abs(val["demand"] - val["pred"])

    # ========================================
    # MAE by hour + count (dual axis)
    # ========================================

    hour_mae = val.groupby("hour")["error"].mean()
    hour_count = val.groupby("hour")["demand"].sum()

    # Save as proper DataFrame with correct column names
    hour_stats = pd.DataFrame({
        "hour": hour_mae.index,
        "mae": hour_mae.values,
        "count": hour_count.values
    })
    hour_stats.to_csv(os.fspath(results / "mae_by_hour.csv"), index=False)

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # MAE left axis
    ax1.plot(hour_mae.index, hour_mae.values,
            color="blue", marker="o", linewidth=2, label="MAE")
    ax1.set_xlabel("Hour")
    ax1.set_ylabel("MAE", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")
    ax1.grid(True, alpha=0.3)

    # Count right axis
    ax2 = ax1.twinx()
    ax2.bar(hour_count.index, hour_count.values,
            color="lightgray", alpha=0.6, label="Count")
    ax2.set_ylabel("Count", color="gray")
    ax2.tick_params(axis="y", labelcolor="gray")

    plt.title("MAE by Hour with Count (Dual Axis)")
    fig.tight_layout()
    plt.savefig(os.fspath(results / "mae_by_hour_dual_axis.png"), dpi=120)
    plt.close()

    # ========================================
    # MAE by borough
    # ========================================
    borough_mae = val.groupby("Borough")["error"].mean()
    borough_mae.to_csv(os.fspath(results / "mae_by_borough.csv"))

    plt.figure(figsize=(10, 5))
    colors_borough = sns.color_palette("viridis", len(borough_mae))
    plt.bar(range(len(borough_mae)), borough_mae.values, color=colors_borough)
    plt.xticks(range(len(borough_mae)), borough_mae.index, rotation=45)
    plt.title("MAE by Borough")
    plt.xlabel("Borough")
    plt.ylabel("MAE")
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.fspath(results / "mae_by_borough.png"), dpi=100)
    plt.close()

    # ========================================
    # MAE by weather (Dry, Rain, Snow)
    # DATA VALIDATION
    # ========================================
    
    # Check data availability
    rain_count = (val["precipitation"] > 0).sum()
    snow_count = (val["snowfall"] > 0).sum()
    dry_count = (val["precipitation"] == 0).sum()
    
    logger.info(f"Weather data counts - Dry: {dry_count}, Rain: {rain_count}, Snow: {snow_count}")
    
    weather_data = []
    
    # Dry condition
    weather_data.append({
        "weather": "Dry",
        "mae": val[val["precipitation"] == 0]["error"].mean(),
        "count": dry_count
    })
    
    # Rain condition
    weather_data.append({
        "weather": "Rain",
        "mae": val[val["precipitation"] > 0]["error"].mean(),
        "count": rain_count
    })
    
    # Snow condition (only if data exists)
    if snow_count > 0:
        snow_mae = val[val["snowfall"] > 0]["error"].mean()
        weather_data.append({
            "weather": "Snow",
            "mae": snow_mae,
            "count": snow_count
        })
        logger.info(f"Snow MAE: {snow_mae:.4f}")
    else:
        logger.warning(f"No snow data in validation set (snowfall > 0 records: {snow_count})")
    
    # Create DataFrame from list of dicts
    weather_df = pd.DataFrame(weather_data)
    weather_df.to_csv(os.fspath(results / "mae_by_weather.csv"), index=False)
    
    # Extract for plotting
    labels = weather_df["weather"].tolist()
    values = weather_df["mae"].tolist()

    plt.figure(figsize=(7, 5))
    colors_weather = sns.color_palette("coolwarm", len(values))
    plt.bar(range(len(values)), values, color=colors_weather)
    plt.xticks(range(len(values)), labels)
    plt.title("MAE: Dry vs Rain vs Snow")
    plt.xlabel("Weather Condition")
    plt.ylabel("MAE")
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.fspath(results / "mae_by_weather.png"), dpi=120)
    plt.close()

    # ========================================
    # Save model
    # ========================================
    joblib.dump(winner_pipe, os.fspath(models_dir / "demand_model.joblib"))

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

    with open(os.fspath(models_dir / "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Winner: {winner_name} (MAE={winner_mae:.4f})")
    logger.info("Model saved.")