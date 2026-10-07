import argparse

from src.utils import load_config, setup_logging
from src.ingest import (
    download_trips, download_zone_lookup, build_interim_trips,
    fetch_weather, build_holidays,
)
from src.validate import validate_all
from src.audit import audit_trips
from src.joins_check import join_readiness
from src.clean import run_clean
from src.features import run_features
from src.train import run_train
from src.predict import run_predict


def run_ingest(config):
    """
    Full ingestion stage:
    1. Download raw data
    2. Build interim tables
    3. Validate all sources
    4. Run data-quality audit
    5. Run join readiness checks
    """
    download_trips(config)
    download_zone_lookup(config)
    build_interim_trips(config)
    fetch_weather(config)
    build_holidays(config)
    validate_all(config)
    audit_trips(config)
    join_readiness(config)


def run_day2(config):
    """
    Day 2: Clean → Features pipeline.
    """
    run_clean(config)
    run_features(config)

STAGES = {
    "ingest": run_ingest,
    "validate": validate_all,
    "audit": audit_trips,
    "joins": join_readiness,
    "clean": run_clean,
    "features": run_day2,
    "train": run_train,
    "predict": lambda config: run_predict(
        config,
        "data/processed/test.parquet",
        "results/predictions.parquet"
    )
}


def main():
    """Entry point: load config, set up logging, run selected stage."""
    config = load_config()
    logger = setup_logging(config)

    parser = argparse.ArgumentParser(description="NYC Taxi data pipeline")
    parser.add_argument("--stage", required=True, choices=list(STAGES))
    args = parser.parse_args()

    logger.info(f"Starting stage: {args.stage}")
    STAGES[args.stage](config)
    logger.info(f"Finished stage: {args.stage}")


if __name__ == "__main__":
    main()
