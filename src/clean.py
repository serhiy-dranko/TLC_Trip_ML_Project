import logging
from pathlib import Path
import pandas as pd

logger = logging.getLogger("pipeline")


# ---------------------------------------------------------
# Helpers: find pickup/dropoff columns
# ---------------------------------------------------------

def find_pickup_column(df):
    for c in df.columns:
        if "pickup_datetime" in c.lower():
            return c
    raise KeyError("No pickup_datetime column found")


def find_dropoff_column(df):
    for c in df.columns:
        if "dropoff_datetime" in c.lower():
            return c
    raise KeyError("No dropoff_datetime column found")


# ---------------------------------------------------------
# Cleaning rule functions
# ---------------------------------------------------------

def drop_pickup_outside_file_month(df, month):
    pickup_col = find_pickup_column(df)
    before = len(df)
    mask = (df[pickup_col].dt.year * 100 + df[pickup_col].dt.month) == int(month.replace("-", ""))
    df = df[mask]
    logger.info(f"drop_pickup_outside_file_month: removed {before - len(df)}")
    return df


def drop_pickup_outside_window(df, window_start, window_end):
    pickup_col = find_pickup_column(df)
    before = len(df)
    mask = (df[pickup_col] >= window_start) & (df[pickup_col] <= window_end)
    df = df[mask]
    logger.info(f"drop_pickup_outside_window: removed {before - len(df)}")
    return df


def drop_dropoff_before_pickup(df):
    pickup_col = find_pickup_column(df)
    dropoff_col = find_dropoff_column(df)
    before = len(df)
    df = df[df[dropoff_col] >= df[pickup_col]]
    logger.info(f"drop_dropoff_before_pickup: removed {before - len(df)}")
    return df


def drop_short_duration(df, min_minutes):
    pickup_col = find_pickup_column(df)
    dropoff_col = find_dropoff_column(df)
    before = len(df)
    duration = (df[dropoff_col] - df[pickup_col]).dt.total_seconds() / 60
    df = df[duration >= min_minutes]
    logger.info(f"drop_short_duration (<{min_minutes} min): removed {before - len(df)}")
    return df


def drop_long_duration(df, max_minutes):
    pickup_col = find_pickup_column(df)
    dropoff_col = find_dropoff_column(df)
    before = len(df)
    duration = (df[dropoff_col] - df[pickup_col]).dt.total_seconds() / 60
    df = df[duration <= max_minutes]
    logger.info(f"drop_long_duration (>{max_minutes} min): removed {before - len(df)}")
    return df


def drop_nonpositive_distance(df):
    if "trip_distance" not in df.columns:
        return df  # FHV has no distance column
    before = len(df)
    df = df[df["trip_distance"] > 0]
    logger.info(f"drop_nonpositive_distance: removed {before - len(df)}")
    return df


def drop_implausible_distance(df, max_miles):
    if "trip_distance" not in df.columns:
        return df
    before = len(df)
    df = df[df["trip_distance"] <= max_miles]
    logger.info(f"drop_implausible_distance (>{max_miles} miles): removed {before - len(df)}")
    return df


def drop_nonpositive_fare(df):
    if "fare_amount" not in df.columns:
        return df
    before = len(df)
    df = df[df["fare_amount"] > 0]
    logger.info(f"drop_nonpositive_fare: removed {before - len(df)}")
    return df


def drop_nonpositive_total(df):
    if "total_amount" not in df.columns:
        return df
    before = len(df)
    df = df[df["total_amount"] > 0]
    logger.info(f"drop_nonpositive_total: removed {before - len(df)}")
    return df


def drop_null_zone(df):
    before = len(df)
    df = df[df["PULocationID"].notna()]
    logger.info(f"drop_null_zone: removed {before - len(df)}")
    return df


def drop_unknown_zones(df):
    before = len(df)
    df = df[~df["PULocationID"].isin([264, 265])]
    logger.info(f"drop_unknown_zones: removed {before - len(df)}")
    return df


def drop_zone_not_in_lookup(df, zone_ids):
    before = len(df)
    df = df[df["PULocationID"].isin(zone_ids)]
    logger.info(f"drop_zone_not_in_lookup: removed {before - len(df)}")
    return df


# ---------------------------------------------------------
# Main cleaning pipeline
# ---------------------------------------------------------

def clean_month(df, source, month, config, zone_ids):
    window = config["data_window"]
    window_start = pd.Timestamp(f"{window['start_month']}-01")
    window_end = pd.Period(window["end_month"], freq="M").end_time

    thresholds = config["cleaning_thresholds"]

    df = drop_pickup_outside_file_month(df, month)
    df = drop_pickup_outside_window(df, window_start, window_end)
    df = drop_dropoff_before_pickup(df)
    df = drop_short_duration(df, thresholds["min_duration_min"])
    df = drop_long_duration(df, thresholds["max_duration_min"])

    # Only apply distance/fare rules if columns exist
    df = drop_nonpositive_distance(df)
    df = drop_implausible_distance(df, thresholds["max_distance_miles"])
    df = drop_nonpositive_fare(df)
    df = drop_nonpositive_total(df)

    df = drop_null_zone(df)
    df = drop_unknown_zones(df)
    df = drop_zone_not_in_lookup(df, zone_ids)

    return df


def run_clean(config):
    interim = Path(config["paths"]["interim"])
    processed = Path(config["paths"]["processed"])
    processed.mkdir(parents=True, exist_ok=True)

    zone_ids = pd.read_parquet(interim / "zone_lookup.parquet")["LocationID"].dropna().tolist()

    rows = []

    from src.ingest import SOURCES, source_months

    for source, (_, columns_key) in SOURCES.items():
        for month in source_months(config, source):
            path = interim / f"trips_{source}_{month}.parquet"
            df = pd.read_parquet(path)

            before = len(df)
            df_clean = clean_month(df, source, month, config, zone_ids)
            after = len(df_clean)

            df_clean.to_parquet(processed / f"cleaned_{source}_{month}.parquet", index=False)

            rows.append({
                "source": source,
                "month": month,
                "rows_before": before,
                "rows_after": after,
                "removed": before - after,
                "pct_removed": round(100 * (before - after) / before, 4)
            })

            logger.info(f"Cleaned {source} {month}: removed {before - after}")

    report = pd.DataFrame(rows)
    out = Path(config["paths"]["results"]) / "cleaning_report.csv"
    report.to_csv(out, index=False)
    logger.info(f"Saved cleaning report -> {out}")
