import logging
from pathlib import Path
import yaml

# Load configuration from YAML file
def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

# Configure logging to both console and file
def setup_logging(config):
    results_dir = Path(config["paths"]["results"])
    results_dir.mkdir(parents=True, exist_ok=True)

    log_path = results_dir / "pipeline.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),       # Console output
            logging.FileHandler(log_path)  # Log file output
        ]
    )
    return logging.getLogger("pipeline")
