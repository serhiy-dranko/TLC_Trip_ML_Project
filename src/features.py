import logging
from pathlib import Path
import pandas as pd
import numpy as np

logger = logging.getLogger("pipeline")


# ---------------------------------------------------------
# 1. Load cleaned trips
# ---------------------------------------------------------

def find_pickup_column(df):
    for c in df.columns:
        if "pickup_datetime" in c.lower():
            return c
    raise KeyError("No pickup_datetime column found")


def load_cleaned_trips(config):
    """Load all cleaned monthly trip files into one DataFrame."""
    processed = Path(config["paths"]["processed"])
    from src.ingest import SOURCES, source_months

    dfs = []
    for source in SOURCES:
        for month in source_months(config, source):
            path = processed / f"cleaned_{source}_{month}.parquet"
            df = pd.read_parquet(path)
            dfs.append(df)

    all_trips = pd.concat(dfs, ignore_index=True)
    logger.info(f"Loaded cleaned trips: {len(all_trips):,} rows")
    return all_trips


# ---------------------------------------------------------
# 2. Aggregate zone × hour
# ---------------------------------------------------------

def aggregate_zone_hour(df):
    """Floor pickup time to hour and count trips per (zone, hour)."""
    pickup_col = find_pickup_column(df)
    df["pickup_hour"] = df[pickup_col].dt.floor("H")
    grouped = df.groupby(["PULocationID", "pickup_hour"]).size().reset_index(name="demand")
    logger.info(f"Aggregated zone-hour rows: {len(grouped):,}")
    return grouped


# ---------------------------------------------------------
# 3. Build complete zone × hour grid
# ---------------------------------------------------------

def build_zone_hour_grid(config, zone_ids):
    """Build complete grid of every zone × every hour in the window."""
    window = config["data_window"]
    start = pd.Timestamp(f"{window['start_month']}-01")
    end = pd.Period(window["end_month"], freq="M").end_time.floor("H")

    hours = pd.date_range(start, end, freq="H")
    grid = pd.MultiIndex.from_product([zone_ids, hours], names=["PULocationID", "pickup_hour"])
    grid = grid.to_frame(index=False)

    logger.info(f"Grid size: {len(grid):,} rows")
    return grid


def join_counts(grid, counts):
    """Left-join counts onto grid and fill missing with 0."""
    df = grid.merge(counts, on=["PULocationID", "pickup_hour"], how="left")
    df["demand"] = df["demand"].fillna(0).astype(int)
    return df


# ---------------------------------------------------------
# 4. Join external data
# ---------------------------------------------------------

def join_zone_lookup(df, config):
    lookup = pd.read_parquet(Path(config["paths"]["interim"]) / "zone_lookup.parquet")
    before = len(df)
    df = df.merge(lookup, left_on="PULocationID", right_on="LocationID", how="left")
    assert len(df) == before, "Row count changed after zone lookup join!"
    return df


def join_weather(df, config):
    weather = pd.read_parquet(Path(config["paths"]["interim"]) / "weather.parquet")
    before = len(df)
    df = df.merge(weather, left_on="pickup_hour", right_on="datetime", how="left")
    assert len(df) == before, "Row count changed after weather join!"
    assert df["temperature_2m"].notna().all(), "Missing weather values!"
    df = df.drop(columns=["datetime"])
    return df


def join_holidays(df, config):
    holidays = pd.read_parquet(Path(config["paths"]["interim"]) / "holidays.parquet")

    # Convert to datetime64[ns] to match holidays table
    df["date"] = df["pickup_hour"].dt.floor("D")

    before = len(df)
    df = df.merge(holidays, on="date", how="left")
    assert len(df) == before, "Row count changed after holiday join!"

    df["is_holiday"] = df["holiday_name"].notna().astype(int)
    return df


# ---------------------------------------------------------
# 5. Feature engineering
# ---------------------------------------------------------

def add_calendar_features(df):
    df["hour"] = df["pickup_hour"].dt.hour
    df["day_of_week"] = df["pickup_hour"].dt.dayofweek
    df["month"] = df["pickup_hour"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    return df


def add_lag_features(df):
    df = df.sort_values(["PULocationID", "pickup_hour"])
    df["lag_1"] = df.groupby("PULocationID")["demand"].shift(1)
    df["lag_24"] = df.groupby("PULocationID")["demand"].shift(24)
    df["lag_168"] = df.groupby("PULocationID")["demand"].shift(168)
    return df


def add_rolling_features(df):
    df = df.sort_values(["PULocationID", "pickup_hour"])
    df["rolling_24"] = (
        df.groupby("PULocationID")["demand"]
        .shift(1)
        .rolling(24)
        .mean()
    )
    return df


def drop_warmup_rows(df):
    before = len(df)
    df = df.dropna(subset=["lag_168", "rolling_24"])
    logger.info(f"Dropped warm-up rows: {before - len(df)}")
    return df


# ---------------------------------------------------------
# 6. Time split
# ---------------------------------------------------------

def time_split(df, config):
    df["month_str"] = df["pickup_hour"].dt.strftime("%Y-%m")

    train = df[df["month_str"].isin(["2026-01", "2026-02", "2026-03", "2026-04"])]
    val = df[df["month_str"] == "2026-05"]
    test = df[df["month_str"] == "2026-06"]

    processed = Path(config["paths"]["processed"])
    train.to_parquet(processed / "train.parquet", index=False)
    val.to_parquet(processed / "val.parquet", index=False)
    test.to_parquet(processed / "test.parquet", index=False)

    logger.info(f"Train rows: {len(train):,}")
    logger.info(f"Val rows: {len(val):,}")
    logger.info(f"Test rows: {len(test):,}")

    return train, val, test


# ---------------------------------------------------------
# 7. Baseline MAE
# ---------------------------------------------------------

def baseline_mae(val):
    y_true = val["demand"]
    y_pred = val["lag_168"]
    mae = np.mean(np.abs(y_true - y_pred))
    logger.info(f"Baseline MAE (lag_168): {mae:.4f}")
    return mae


# ---------------------------------------------------------
# 8. Main pipeline
# ---------------------------------------------------------

def run_features(config):
    interim = Path(config["paths"]["interim"])
    zone_ids = pd.read_parquet(interim / "zone_lookup.parquet")["LocationID"].dropna().tolist()

    trips = load_cleaned_trips(config)
    counts = aggregate_zone_hour(trips)
    grid = build_zone_hour_grid(config, zone_ids)
    df = join_counts(grid, counts)

    df = join_zone_lookup(df, config)
    df = join_weather(df, config)
    df = join_holidays(df, config)

    df = add_calendar_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = drop_warmup_rows(df)

    train, val, test = time_split(df, config)
    mae = baseline_mae(val)

    with open("decision_log.md", "a", encoding="utf-8") as f:
        f.write(f"\n\n## Day 2 Baseline\nBaseline MAE (lag_168): {mae:.4f}\n")

    logger.info("Day-2 features pipeline completed.")
