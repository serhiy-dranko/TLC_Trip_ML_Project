import pandas as pd
import joblib
from pathlib import Path
import json


def load_model(models_dir):
    model_path = Path(models_dir) / "demand_model.joblib"
    metadata_path = Path(models_dir) / "model_metadata.json"

    model = joblib.load(model_path)

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    return model, metadata


def run_predict(config, input_path, output_path):
    models_dir = config["paths"]["models"]

    # Load model + metadata
    model, metadata = load_model(models_dir)

    # Load input data
    df = pd.read_parquet(input_path)

    # Ensure required columns exist
    missing = [col for col in metadata["features"] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Predict
    preds = model.predict(df[metadata["features"]])
    preds = preds.clip(0)  # no negative demand

    df["predicted_demand"] = preds

    # Save output
    df.to_parquet(output_path, index=False)

    print(f"Saved predictions to: {output_path}")
