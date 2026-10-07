import logging
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger("pipeline")


class ValidationError(Exception):
    """Raised when a validation check fails."""


# Local copy of SOURCES to avoid circular import
SOURCES = {
    "yellow": ("yellow_trips_base", "yellow_trip_columns"),
    "green": ("green_trips_base", "green_trip_columns"),
    "fhv": ("fhv_trips_base", "fhv_trip_columns"),
}


def source_months(config, source):
    """Generate list of months for each source."""
    window = config["data_window"]
    end = window.get("source_end_month", {}).get(source, window["end_month"])
    start = window["start_month"]

    period = pd.Period(start, freq="M")
    end_p = pd.Period(end, freq="M")

    months = []
    while period <= end_p:
        months.append(str(period))
        period += 1
    return months


def _fail(msg):
    raise ValidationError(msg)


def check_columns(actual, expected, label):
    """Ensure all expected columns exist."""
    missing = [c for c in expected if c not in list(actual)]
    if missing:
        _fail(f"{label}: missing columns {missing}. Found: {list(actual)}")


def pick_col(columns, fragment):
    """Find column containing fragment (case-insensitive)."""
    return next(c for c in columns if fragment in c.lower())


def validate_trip_row_counts(counts, source, low=0.5, high=2.0):
    """Check monthly row counts for anomalies."""
    median = pd.Series(counts).median()
    bad = {m: n for m, n in counts.items() if not (low * median <= n <= high * median)}
    if bad:
        _fail(f"{source}: row counts far from median ({median:,.0f}): {bad}")


def validate_trip_file(path, columns, source):
    """
    Validate a single Parquet trip file:
    - schema check
    - timestamp type check
    - row count check
    - null pickup_datetime check
    - null PULocationID check (yellow/green strict, fhv lenient)
    """
    label = Path(path).name

    # 1. Schema
    schema = pq.read_schema(path)
    check_columns(schema.names, columns, label)

    # Check timestamp types
    for c in schema.names:
        if "datetime" in c.lower() and not pa.types.is_timestamp(schema.field(c).type):
            _fail(f"{label}: column {c} is not a timestamp (type {schema.field(c).type})")

    # 2. Row count
    n_rows = pq.ParquetFile(path).metadata.num_rows
    if n_rows == 0:
        _fail(f"{label}: file has 0 rows")

    # 3. Null checks
    pickup = pick_col(columns, "pickup_datetime")
    zone = pick_col(columns, "pulocationid")

    df = pd.read_parquet(path, columns=[pickup, zone])

    if df[pickup].isna().any():
        _fail(f"{label}: {int(df[pickup].isna().sum())} null values in {pickup}")

    zone_nulls = int(df[zone].isna().sum())

    if source == "fhv":
        if zone_nulls:
            logger.warning(f"{label}: {zone_nulls} null {zone} ({zone_nulls / n_rows:.1%})")
    else:
        if zone_nulls:
            _fail(f"{label}: {zone_nulls} null values in {zone}")

    return n_rows


def validate_trips_month(df, source, month, config):
    """Validate a single month of trips BEFORE writing to interim."""
    if len(df) == 0:
        raise ValidationError(f"{source} {month}: no rows after loading")

    pickup_col = next(c for c in df.columns if "pickup_datetime" in c.lower())
    dropoff_col = next(c for c in df.columns if "dropoff_datetime" in c.lower())

    if not pd.api.types.is_datetime64_any_dtype(df[pickup_col]):
        raise ValidationError(f"{source} {month}: {pickup_col} is not datetime")
    if not pd.api.types.is_datetime64_any_dtype(df[dropoff_col]):
        raise ValidationError(f"{source} {month}: {dropoff_col} is not datetime")

    if df[pickup_col].isna().any():
        raise ValidationError(
            f"{source} {month}: {df[pickup_col].isna().sum()} null pickup timestamps"
        )

    zone_col = next(c for c in df.columns if c.lower() == "pulocationid")
    zone_nulls = df[zone_col].isna().sum()

    if source == "fhv":
        if zone_nulls:
            logger.warning(
                f"{source} {month}: {zone_nulls} null PULocationID ({zone_nulls/len(df):.1%})"
            )
    else:
        if zone_nulls:
            raise ValidationError(
                f"{source} {month}: {zone_nulls} null PULocationID"
            )

    return True


def validate_zone_lookup(df):
    """Validate zone lookup table."""
    check_columns(df.columns, ["LocationID", "Borough", "Zone"], "zone lookup")
    if df["LocationID"].duplicated().any():
        _fail("zone lookup: LocationID is not unique")
    logger.info("Zone lookup OK")


def _window(config):
    """Return start and end timestamps for weather window."""
    w = config["data_window"]
    start = pd.Timestamp(f"{w['start_month']}-01")
    end = pd.Period(w["end_month"], freq="M").end_time.floor("h")
    return start, end


def validate_weather(df, config, time_col="datetime"):
    """Validate weather table."""
    hourly = config["weather"]["hourly"]
    check_columns(df.columns, [time_col] + hourly, "weather")

    start, end = _window(config)
    tz = config["weather"]["timezone"]

    naive_hours = len(pd.date_range(start, end, freq="h"))
    real_hours = int((pd.Timestamp(end + pd.Timedelta("1h")).tz_localize(tz)
                      - start.tz_localize(tz)) / pd.Timedelta("1h"))

    n = len(df)
    if n not in (naive_hours, real_hours):
        _fail(f"weather: {n} rows, expected {naive_hours} or {real_hours}")

    if df[time_col].duplicated().any():
        _fail("weather: duplicate timestamps")

    if "temperature_2m" in df and not df["temperature_2m"].between(-30, 45).all():
        _fail("weather: temperature_2m outside -30..45 C")

    logger.info("Weather OK")


def validate_holidays(df, config):
    """Validate holiday table."""
    start, end = _window(config)
    outside = df[(df["date"] < start) | (df["date"] > end)]
    if len(outside):
        _fail("holidays: dates outside window")
    logger.info("Holidays OK")


def validate_all(config):
    """Run all validation steps."""
    interim = Path(config["paths"]["interim"])

    # Trips
    for source, (_, columns_key) in SOURCES.items():
        counts = {}
        for month in source_months(config, source):
            path = interim / f"trips_{source}_{month}.parquet"
            counts[month] = validate_trip_file(path, config[columns_key], source)
        validate_trip_row_counts(counts, source)

    # Zone lookup
    validate_zone_lookup(pd.read_parquet(interim / "zone_lookup.parquet"))

    # Weather
    validate_weather(pd.read_parquet(interim / "weather.parquet"), config)

    # Holidays
    validate_holidays(pd.read_parquet(interim / "holidays.parquet"), config)

    logger.info("All validation checks passed")
