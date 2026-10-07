import logging
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from src.ingest import SOURCES, source_months
from src.validate import _window

logger = logging.getLogger("pipeline")

# Join plan text to be written into decision_log.md
JOIN_PLAN = """
## Join plan
- trips -> zone lookup: trip grain (1 row per trip) vs zone grain (1 row per LocationID); key PULocationID = LocationID; many_to_one, row count unchanged.
- (zone, hour) demand -> weather: (zone, hour) grain vs hour grain; key pickup_hour = weather datetime; many_to_one, row count unchanged (= number of zone-hour rows).
- (zone, hour) demand -> holidays: (zone, hour) grain vs date grain; key date(pickup_hour) = holiday date; many_to_one, row count unchanged; non-holidays get null -> is_holiday flag.
- Weather is joined AFTER aggregating trips to (zone, hour): joining at trip level would repeat the same hourly values millions of times.
"""


def join_readiness(config):
    """
    Check join readiness:
    - coverage of pickup zones in zone lookup
    - weather coverage over the window
    - clock-change behavior
    - write summary + join plan to results/join_readiness.md and decision_log.md
    """
    interim = Path(config["paths"]["interim"])
    lines = ["# Join readiness"]

    # 1. Pickup zone coverage per source
    zone_ids = pd.read_parquet(interim / "zone_lookup.parquet")["LocationID"].dropna().tolist()
    lines.append("\n## Pickup zone found in lookup")
    for source in SOURCES:
        total = nonnull = found = 0
        for month in source_months(config, source):
            path = interim / f"trips_{source}_{month}.parquet"
            col = next(c for c in pq.read_schema(path).names if c.lower() == "pulocationid")
            pu = pd.read_parquet(path, columns=[col])[col]
            total += len(pu)
            nonnull += int(pu.notna().sum())
            found += int(pu.isin(zone_ids).sum())
        lines.append(
            f"- {source}: {100 * found / total:.2f}% of all rows, "
            f"{100 * found / max(nonnull, 1):.2f}% of non-null rows ({total:,} rows)"
        )

    # 2. Weather coverage
    weather = pd.read_parquet(interim / "weather.parquet")
    start, end = _window(config)
    expected = pd.date_range(start, end, freq="h")
    stamps = pd.DatetimeIndex(weather["datetime"])
    missing = expected.difference(stamps)
    extra = stamps.difference(expected)
    dups = int(weather["datetime"].duplicated().sum())

    lines.append("\n## Weather coverage")
    lines.append(f"- expected {len(expected)} hourly rows (24/day), found {len(weather)}")
    lines.append(f"- missing hours: {len(missing)} {list(missing[:10].astype(str))}")
    lines.append(f"- unexpected hours: {len(extra)}, duplicated timestamps: {dups}")

    # 3. Clock changes (DST transitions)
    tz = config["weather"]["timezone"]
    days = pd.date_range(start.normalize(), end.normalize(), freq="D")
    offsets = [d.tz_localize(tz).utcoffset() for d in days]
    lines.append("\n## Clock changes")
    transitions = [days[i - 1] for i in range(1, len(days)) if offsets[i] != offsets[i - 1]]
    if not transitions:
        lines.append("- none in this window")
    for day in transitions:
        hours = sorted(
            weather.loc[weather["datetime"].dt.normalize() == day, "datetime"].dt.hour.tolist()
        )
        lines.append(f"- {day.date()}: weather has {len(hours)} rows; hours present: {hours}")

    text = "\n".join(lines) + "\n" + JOIN_PLAN
    out = Path(config["paths"]["results"]) / "join_readiness.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")

    # Append join plan to decision_log.md only once
    log_path = Path("decision_log.md")
    existing = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
    if "## Join plan" not in existing:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write("\n" + JOIN_PLAN)

    logger.info(f"Saved join readiness report -> {out}")
    print(text)
