import logging
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from src.ingest import SOURCES, source_months

logger = logging.getLogger("pipeline")

# Thresholds live here so the audit and tomorrow's cleaning rules use the same numbers
MIN_DURATION_MIN = 1
MAX_DURATION_MIN = 180
MAX_DISTANCE_MILES = 100

# One-sentence proposed rule per issue (keys must match the issue names below)
RULES = {
    "pickup outside file month": "Drop rows whose pickup month differs from the file's month.",
    "pickup outside window": "Drop rows with pickup outside the configured window.",
    "dropoff before pickup": "Drop rows where dropoff is earlier than pickup.",
    f"duration under {MIN_DURATION_MIN} min (non-negative)": "Drop trips shorter than 1 minute.",
    f"duration over {MAX_DURATION_MIN} min": "Drop trips longer than 3 hours.",
    "distance <= 0": "Drop trips with non-positive distance.",
    f"distance > {MAX_DISTANCE_MILES} miles": "Drop trips above 100 miles as implausible for NYC taxis.",
    "fare_amount <= 0": "Drop trips with non-positive fare.",
    "total_amount <= 0": "Drop trips with non-positive total amount.",
    "passenger_count null": "Keep rows; passenger_count is not used for demand counts.",
    "PULocationID null": "Drop rows without a pickup zone (cannot be assigned to a zone-hour).",
    "PULocationID in (264, 265)": "Drop unknown / outside-NYC pickup zones.",
    "PULocationID not in zone lookup": "Drop rows whose pickup zone is not in the lookup.",
}


def _get(df, cols, name):
    """Case-insensitive column access; returns None if column is missing."""
    return df[cols[name]] if name in cols else None


def _month_issues(df, month, window_start, window_end, zone_ids):
    """
    Count rows with each problem in one monthly table.
    Nothing is modified or deleted; this is pure measurement.
    """
    cols = {c.lower(): c for c in df.columns}
    pick = df[next(c for c in df.columns if "pickup_datetime" in c.lower())]
    drop = df[next(c for c in df.columns if "dropoff_datetime" in c.lower())]
    dur = (drop - pick).dt.total_seconds() / 60   # trip duration in minutes

    issues = {
        "pickup outside file month": (pick.dt.year * 100 + pick.dt.month) != int(month.replace("-", "")),
        "pickup outside window": (pick < window_start) | (pick > window_end),
        "dropoff before pickup": drop < pick,
        f"duration under {MIN_DURATION_MIN} min (non-negative)": (dur >= 0) & (dur < MIN_DURATION_MIN),
        f"duration over {MAX_DURATION_MIN} min": dur > MAX_DURATION_MIN,
    }

    dist = _get(df, cols, "trip_distance")
    if dist is not None:
        issues["distance <= 0"] = dist <= 0
        issues[f"distance > {MAX_DISTANCE_MILES} miles"] = dist > MAX_DISTANCE_MILES

    fare = _get(df, cols, "fare_amount")
    if fare is not None:
        issues["fare_amount <= 0"] = fare <= 0

    total = _get(df, cols, "total_amount")
    if total is not None:
        issues["total_amount <= 0"] = total <= 0

    pax = _get(df, cols, "passenger_count")
    if pax is not None:
        issues["passenger_count null"] = pax.isna()

    pu = _get(df, cols, "pulocationid")
    if pu is not None:
        issues["PULocationID null"] = pu.isna()
        issues["PULocationID in (264, 265)"] = pu.isin([264, 265])
        issues["PULocationID not in zone lookup"] = pu.notna() & ~pu.isin(zone_ids)

    return {name: int(mask.sum()) for name, mask in issues.items()}


def audit_schema(config):
    """
    Detect schema differences between months:
    - columns present only in some months (e.g. cbd_congestion_fee)
    - columns with different casing across months (e.g. airport_fee)
    """
    rows = []
    raw = Path(config["paths"]["raw"])
    for source in SOURCES:
        months = source_months(config, source)
        seen = {}
        for m in months:
            for name in pq.read_schema(raw / f"{source}_tripdata_{m}.parquet").names:
                seen.setdefault(name.lower(), {}).setdefault(name, []).append(m)
        for low, variants in seen.items():
            present = sum(len(v) for v in variants.values())
            if present < len(months):
                rows.append({
                    "source": source,
                    "issue": f"column '{low}' present in {present} of {len(months)} monthly files",
                    "count": len(months) - present,
                    "pct_of_rows": None,
                    "proposed_rule": "Select by name; fill with null or exclude if absent in some months.",
                })
            if len(variants) > 1:
                rows.append({
                    "source": source,
                    "issue": f"column '{low}' spelled differently: {sorted(variants)}",
                    "count": len(months) - max(len(v) for v in variants.values()),
                    "pct_of_rows": None,
                    "proposed_rule": "Match column names case-insensitively and rename to one spelling.",
                })
    return rows


def audit_trips(config):
    """
    Run data-quality audit for all trip sources and save results to CSV.
    Does not modify data; only measures issues and proposes cleaning rules.
    """
    interim = Path(config["paths"]["interim"])
    zone_ids = pd.read_parquet(interim / "zone_lookup.parquet")["LocationID"].dropna().tolist()
    window_start = pd.Timestamp(f"{config['data_window']['start_month']}-01")

    rows = []
    for source in SOURCES:
        months = source_months(config, source)
        window_end = pd.Period(months[-1], freq="M").end_time
        totals, n_rows = {}, 0

        for month in months:
            df = pd.read_parquet(interim / f"trips_{source}_{month}.parquet")
            n_rows += len(df)
            for issue, count in _month_issues(df, month, window_start, window_end, zone_ids).items():
                totals[issue] = totals.get(issue, 0) + count
            logger.info(f"Audited {source} {month}")

        for issue, count in totals.items():
            rows.append({
                "source": source,
                "issue": issue,
                "count": count,
                "pct_of_rows": round(100 * count / n_rows, 4),
                "proposed_rule": RULES.get(issue, ""),
            })

    rows += audit_schema(config)
    audit = pd.DataFrame(rows)
    out = Path(config["paths"]["results"]) / "data_audit.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(out, index=False)
    logger.info(f"Saved audit with {len(audit)} rows -> {out}")
    return audit
