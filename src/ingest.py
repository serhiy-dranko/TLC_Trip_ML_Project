import json
import logging
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import requests

from src.validate import (
    validate_trips_month, validate_trip_row_counts, validate_zone_lookup,
    validate_weather, validate_holidays,
)

logger = logging.getLogger("pipeline")

# Mapping of data sources to config keys
SOURCES = {
    "yellow": ("yellow_trips_base", "yellow_trip_columns"),
    "green": ("green_trips_base", "green_trip_columns"),
    "fhv": ("fhv_trips_base", "fhv_trip_columns"),
}


def month_range(start_month, end_month):
    """Generate YYYY-MM strings between start and end month."""
    period = pd.Period(start_month, freq="M")
    end = pd.Period(end_month, freq="M")
    months = []
    while period <= end:
        months.append(str(period))
        period += 1
    return months


def source_months(config, source):
    """Determine months for each source (FHV may end earlier)."""
    window = config["data_window"]
    end = window.get("source_end_month", {}).get(source, window["end_month"])
    return month_range(window["start_month"], end)


def _download(url, final_path, label):
    """Idempotent download with .tmp file safety."""
    final_path = Path(final_path)
    temp_path = final_path.with_name(final_path.name + ".tmp")

    if final_path.exists():
        logger.info(f"Skipping {label}: already exists")
        return

    logger.info(f"Downloading {label} from {url}")
    try:
        with requests.get(url, stream=True, timeout=60) as response:
            response.raise_for_status()
            with open(temp_path, "wb") as f:
                for chunk in response.iter_content(1024 * 1024):
                    f.write(chunk)
        temp_path.replace(final_path)
    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        raise RuntimeError(f"Failed to download {label}: {e}")
    logger.info(f"Downloaded {label}")


def download_trips(config):
    """Download yellow, green, and FHV trip files."""
    raw_dir = Path(config["paths"]["raw"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    for source, (url_key, _) in SOURCES.items():
        for month in source_months(config, source):
            url = config["urls"][url_key].format(month=month)
            path = raw_dir / f"{source}_tripdata_{month}.parquet"
            _download(url, path, f"{source} {month}")


def download_zone_lookup(config):
    """Download zone lookup CSV and convert to Parquet."""
    raw_dir = Path(config["paths"]["raw"])
    interim_dir = Path(config["paths"]["interim"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    interim_dir.mkdir(parents=True, exist_ok=True)

    raw_csv = raw_dir / "taxi_zone_lookup.csv"
    _download(config["urls"]["zone_lookup"], raw_csv, "zone lookup")

    df = pd.read_csv(raw_csv)
    validate_zone_lookup(df)
    df.to_parquet(interim_dir / "zone_lookup.parquet", index=False)
    logger.info("Saved zone lookup parquet")


def load_trips_month(path, columns):
    """Load only required columns, fix types, rename to config spelling."""
    file_cols = pq.read_schema(path).names
    lookup = {c.lower(): c for c in file_cols}

    missing = [c for c in columns if c.lower() not in lookup]
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}")

    actual = [lookup[c.lower()] for c in columns]
    df = pd.read_parquet(path, columns=actual)
    df = df.rename(columns=dict(zip(actual, columns)))

    for c in columns:
        if "datetime" in c.lower():
            df[c] = pd.to_datetime(df[c])
        elif c.lower() in ("pulocationid", "dolocationid"):
            df[c] = df[c].astype("Int64")

    return df


def build_interim_trips(config):
    """Create interim trip Parquet files with validation."""
    interim_dir = Path(config["paths"]["interim"])
    interim_dir.mkdir(parents=True, exist_ok=True)

    for source, (_, columns_key) in SOURCES.items():
        columns = config[columns_key]
        months = source_months(config, source)
        raw_paths = {m: Path(config["paths"]["raw"]) / f"{source}_tripdata_{m}.parquet" for m in months}

        counts = {m: pq.ParquetFile(p).metadata.num_rows for m, p in raw_paths.items()}
        validate_trip_row_counts(counts, source)

        for month in months:
            df = load_trips_month(raw_paths[month], columns)
            validate_trips_month(df, source, month, config)
            out_path = interim_dir / f"trips_{source}_{month}.parquet"
            df.to_parquet(out_path, index=False)
            logger.info(f"Saved interim {source} {month}")


def fetch_weather(config):
    """Fetch hourly weather and save raw JSON + tidy Parquet."""
    raw_dir = Path(config["paths"]["raw"])
    interim_dir = Path(config["paths"]["interim"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    interim_dir.mkdir(parents=True, exist_ok=True)

    window = config["data_window"]
    start_date = f"{window['start_month']}-01"
    end_date = str(pd.Period(window["end_month"], freq="M").end_time.date())

    w = config["weather"]
    params = {
        "latitude": w["latitude"],
        "longitude": w["longitude"],
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(w["hourly"]),
        "timezone": w["timezone"],
    }

    logger.info("Fetching weather")
    response = requests.get("https://archive-api.open-meteo.com/v1/archive", params=params)
    response.raise_for_status()
    data = response.json()

    with open(raw_dir / "weather_raw.json", "w") as f:
        json.dump(data, f)

    df = pd.DataFrame(data["hourly"]).rename(columns={"time": "datetime"})
    df["datetime"] = pd.to_datetime(df["datetime"])

    validate_weather(df, config)
    df.to_parquet(interim_dir / "weather.parquet", index=False)
    logger.info("Saved weather parquet")


def build_holidays(config):
    """Build holiday table for NY state."""
    import holidays as holidays_pkg

    interim_dir = Path(config["paths"]["interim"])
    interim_dir.mkdir(parents=True, exist_ok=True)

    window = config["data_window"]
    start = pd.Timestamp(f"{window['start_month']}-01")
    end = pd.Period(window["end_month"], freq="M").end_time
    years = range(start.year, end.year + 1)

    ny = holidays_pkg.US(state="NY", years=years)
    records = [
        {"date": pd.Timestamp(d), "holiday_name": name}
        for d, name in sorted(ny.items())
        if start <= pd.Timestamp(d) <= end
    ]

    df = pd.DataFrame(records)
    validate_holidays(df, config)
    df.to_parquet(interim_dir / "holidays.parquet", index=False)
    logger.info("Saved holidays parquet")
