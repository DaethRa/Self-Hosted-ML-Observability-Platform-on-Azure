import json
import os
import threading
from datetime import datetime, timezone

import pandas as pd


EVENTS_PATH = os.getenv(
    "PREDICTION_EVENTS_PATH",
    "/app/monitoring/prediction_events.jsonl"
)

REFERENCE_PATH = os.getenv(
    "REFERENCE_DATA_PATH",
    "/app/models/reference.parquet"
)

_write_lock = threading.Lock()


def extract_features(pipeline, df: pd.DataFrame) -> dict:
    """Extract features using the fitted production enricher."""

    enricher = pipeline.named_steps["enricher"]
    enriched = enricher.transform(df)

    reference_columns = pd.read_parquet(
        REFERENCE_PATH
    ).columns.tolist()

    missing = set(reference_columns) - set(enriched.columns)

    if missing:
        raise ValueError(
            f"Missing monitoring features: {sorted(missing)}"
        )

    features = enriched[reference_columns].iloc[0]

    return {
        key: None if pd.isna(value) else (
            value.item() if hasattr(value, "item") else value
        )
        for key, value in features.items()
    }


def log_prediction(order_id: str, features: dict, result: dict):
    """Append a successful prediction event to JSONL."""

    event = {
        "order_id": order_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "features": features,
        "probability": result["probability"],
        "is_bad_review": result["is_bad_review"],
        "model_version": result["model_version"],
    }

    os.makedirs(os.path.dirname(EVENTS_PATH), exist_ok=True)

    with _write_lock:
        with open(EVENTS_PATH, "a", encoding="utf-8") as file:
            file.write(json.dumps(event, allow_nan=False) + "\n")

def extract_features_from_enriched(enriched: pd.DataFrame) -> dict:
    reference_columns = pd.read_parquet(
        REFERENCE_PATH
    ).columns.tolist()

    missing = set(reference_columns) - set(enriched.columns)

    if missing:
        raise ValueError(
            f"Missing monitoring features: {sorted(missing)}"
        )

    features = enriched[reference_columns].iloc[0]

    return {
        key: None if pd.isna(value) else (
            value.item() if hasattr(value, "item") else value
        )
        for key, value in features.items()
    }