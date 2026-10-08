"""
NYC Taxi Demand Forecasting Pipeline
====================================

One entry point for every stage of the project.

Usage:
  python run_pipeline.py --stage all        # ingest -> clean -> features -> train
  python run_pipeline.py --stage <name>     # run a single stage
  python run_pipeline.py --stage test       # final test evaluation (run ONCE)

Stages:
  ingest    download raw data, build interim tables, validate, audit, check joins
  validate  validation checks only
  audit     data-quality audit only
  joins     join-readiness report only
  clean     apply cleaning rules, write the cleaning report
  features  build zone-hour features, write train / val / test parquet files
  train     compare models, pick the winner, save model, metadata and error breakdowns
  test      final evaluation on the locked test month (June); never re-run after seeing it
  predict   write predictions for the test file

Rules:
  - Stages run in a fixed order and a failed stage stops the run (exit code 1).
  - 'test' is NOT part of 'all' on purpose: the test set is opened once.

Author: Serhiy Dranko
Date: 2026-10-08
"""

import argparse
import sys
import time
from datetime import datetime

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
from src.evaluate import run_test
from src.predict import run_predict


# ============================================================================
# Stage definitions
# ============================================================================

def run_ingest(config):
    """
    Full ingestion stage:
      1. Download raw trips and the zone lookup
      2. Build interim parquet tables, fetch weather, build holidays
      3. Validate all sources
      4. Audit data quality (results/data_audit.csv)
      5. Check join readiness (results/join_readiness.md)
    """
    download_trips(config)
    download_zone_lookup(config)
    build_interim_trips(config)
    fetch_weather(config)
    build_holidays(config)
    validate_all(config)
    audit_trips(config)
    join_readiness(config)


def run_predict_stage(config):
    """Write predictions for the test file (does not compute test MAE)."""
    run_predict(
        config,
        "data/processed/test.parquet",
        "results/predictions.parquet",
    )


STAGES = {
    "ingest": run_ingest,
    "validate": validate_all,
    "audit": audit_trips,
    "joins": join_readiness,
    "clean": run_clean,
    "features": run_features,
    "train": run_train,
    "test": run_test,
    "predict": run_predict_stage,
}

# Order for '--stage all'. 'test' and 'predict' touch the locked test set,
# so they are run separately and deliberately.

PIPELINE_ORDER = ["ingest", "clean", "features", "train", "predict"]

# ============================================================================
# Execution
# ============================================================================

def run_stage(name, config, logger):
    """
    Run one stage with timing and error handling.

    On failure: log the stage name and the full traceback, then exit with code 1.
    Nothing is swallowed and the pipeline never continues with broken inputs.
    """
    logger.info("-" * 60)
    logger.info(f"Starting stage: {name}")
    start = time.perf_counter()

    try:
        STAGES[name](config)
    except Exception:
        elapsed = time.perf_counter() - start
        logger.exception(
            f"Stage '{name}' FAILED after {elapsed:.1f}s. "
            "Check: outputs of earlier stages, config.yaml, and the traceback below."
        )
        sys.exit(1)

    logger.info(f"Finished stage: {name} ({time.perf_counter() - start:.1f}s)")


def run_all(config, logger):
    """Run PIPELINE_ORDER; the first failing stage stops the run."""
    logger.info("=" * 60)
    logger.info("FULL PIPELINE: " + " -> ".join(PIPELINE_ORDER))
    logger.info("=" * 60)

    start = time.perf_counter()
    for i, name in enumerate(PIPELINE_ORDER, 1):
        logger.info(f"[{i}/{len(PIPELINE_ORDER)}] {name}")
        run_stage(name, config, logger)

    logger.info("=" * 60)
    logger.info(f"PIPELINE COMPLETED in {time.perf_counter() - start:.1f}s")
    logger.info("=" * 60)


def main():
    """Entry point: parse arguments, load config, set up logging, run."""
    parser = argparse.ArgumentParser(
        description="NYC Taxi Demand Forecasting Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python run_pipeline.py --stage all\n"
            "  python run_pipeline.py --stage train\n"
            "  python run_pipeline.py --stage test   (run once)\n"
        ),
    )
    parser.add_argument(
        "--stage",
        required=True,
        choices=["all", *STAGES],
        help="stage to run, or 'all' for ingest -> clean -> features -> train",
    )
    args = parser.parse_args()

    config = load_config()
    logger = setup_logging(config)

    logger.info(f"Pipeline started: {datetime.now():%Y-%m-%d %H:%M:%S}, stage = {args.stage}")

    if args.stage == "all":
        run_all(config, logger)
    else:
        run_stage(args.stage, config, logger)

    logger.info(f"Pipeline finished: {datetime.now():%Y-%m-%d %H:%M:%S}")


if __name__ == "__main__":
    main()