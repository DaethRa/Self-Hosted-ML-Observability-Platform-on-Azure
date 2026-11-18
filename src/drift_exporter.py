import json
import os
import time
import logging
from src.performance import update_performance_metrics

import numpy as np
import pandas as pd
from prometheus_client import Gauge, start_http_server

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("drift_exporter")

REFERENCE_PATH = os.getenv(
    "REFERENCE_DATA_PATH", "/app/models/reference.parquet"
)
EVENTS_PATH = os.getenv(
    "PREDICTION_EVENTS_PATH",
    "/app/monitoring/prediction_events.jsonl"
)

MIN_SAMPLES = int(os.getenv("DRIFT_MIN_SAMPLES", "100"))
WINDOW_SIZE = int(os.getenv("DRIFT_WINDOW_SIZE", "1000"))
INTERVAL = int(os.getenv("DRIFT_INTERVAL", "60"))

DRIFT_SCORE = Gauge(
    "ml_feature_drift_score",
    "Feature drift score (PSI numeric, TVD categorical)",
    ["feature", "method"]
)

MISSING_RATE = Gauge(
    "ml_feature_missing_rate",
    "Missing value rate in production",
    ["feature"]
)

SAMPLE_COUNT = Gauge(
    "ml_drift_sample_count",
    "Number of unique orders in current monitoring window"
)

DRIFT_READY = Gauge(
    "ml_drift_ready",
    "Whether enough observations exist to calculate drift"
)

LAST_SUCCESS = Gauge(
    "ml_drift_last_success_timestamp_seconds",
    "Unix timestamp of last successful drift calculation"
)


def calculate_psi(reference, current, bins=10):
    """Population Stability Index with reference-based bins."""
    ref = pd.to_numeric(reference, errors="coerce").dropna()
    cur = pd.to_numeric(current, errors="coerce").dropna()

    if len(ref) == 0 or len(cur) == 0:
        return float("nan")

    edges = np.unique(
        np.quantile(ref, np.linspace(0, 1, bins + 1))
    )

    if len(edges) < 2:
        value = ref.iloc[0]
        return 0.0 if np.allclose(cur, value) else 1.0

    edges[0] = -np.inf
    edges[-1] = np.inf

    ref_counts = np.histogram(ref, bins=edges)[0]
    cur_counts = np.histogram(cur, bins=edges)[0]

    eps = 1e-6

    ref_pct = (ref_counts + eps) / (
        ref_counts.sum() + eps * len(ref_counts)
    )
    cur_pct = (cur_counts + eps) / (
        cur_counts.sum() + eps * len(cur_counts)
    )

    return float(
        np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    )


def calculate_tvd(reference, current):
    """Total Variation Distance for categorical features."""
    ref = reference.fillna("__MISSING__").astype(str)
    cur = current.fillna("__MISSING__").astype(str)

    ref_dist = ref.value_counts(normalize=True)
    cur_dist = cur.value_counts(normalize=True)

    categories = ref_dist.index.union(cur_dist.index)

    return float(
        0.5 * np.abs(
            ref_dist.reindex(categories, fill_value=0)
            - cur_dist.reindex(categories, fill_value=0)
        ).sum()
    )


def load_events():
    if not os.path.exists(EVENTS_PATH):
        return pd.DataFrame()

    records = []

    with open(EVENTS_PATH, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                event = json.loads(line)
                records.append(event)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed event")

    if not records:
        return pd.DataFrame()

    events = pd.DataFrame(records)
    events = events.dropna(subset=["order_id"])

    events = events.drop_duplicates(
        subset=["order_id"], keep="last"
    )

    if "timestamp" in events.columns:
        events = events.sort_values("timestamp")

    return events.tail(WINDOW_SIZE)


def update_metrics(reference):
    events = load_events()
    count = len(events)

    SAMPLE_COUNT.set(count)
    DRIFT_READY.set(0)

    if count < MIN_SAMPLES:
        logger.info(
            "Not enough unique orders: %d/%d",
            count, MIN_SAMPLES
        )
        return

    production = pd.DataFrame(events["features"].tolist())

    scores = {}

    for feature in reference.columns:
        if feature not in production.columns:
            logger.warning("Missing feature: %s", feature)
            continue

        ref_col = reference[feature]
        cur_col = production[feature]

        MISSING_RATE.labels(feature=feature).set(
            float(cur_col.isna().mean())
        )

        if pd.api.types.is_numeric_dtype(ref_col):
            score = calculate_psi(ref_col, cur_col)
            method = "psi"
        else:
            score = calculate_tvd(ref_col, cur_col)
            method = "tvd"

        if np.isfinite(score):
            DRIFT_SCORE.labels(
                feature=feature,
                method=method
            ).set(score)

            scores[feature] = round(score, 4)

    DRIFT_READY.set(1)
    LAST_SUCCESS.set(time.time())

    logger.info(
        "Drift calculated for %d features, %d orders",
        len(scores), count
    )
    logger.info("Scores: %s", scores)


def main():
    reference = pd.read_parquet(REFERENCE_PATH)

    logger.info(
        "Loaded reference: %d rows, %d features",
        len(reference), len(reference.columns)
    )

    start_http_server(8001, addr="0.0.0.0")
    logger.info("Exporter listening on port 8001")

    while True:
        try:
            update_metrics(reference)
        except Exception:
            DRIFT_READY.set(0)
            logger.exception("Drift calculation failed")

        try:
            events = load_events()
            update_performance_metrics(events)
        except Exception:
            logger.exception("Performance calculation failed")

        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
