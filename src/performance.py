import os
import logging
import pandas as pd
from prometheus_client import Gauge
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
)

logger = logging.getLogger("model_performance")

LABELS_PATH = os.getenv(
    "LABELS_PATH", "/app/models/labels.parquet"
)

# Virtual time: only labels available by this date are visible.
SIMULATED_DATE = os.getenv(
    "SIMULATED_LABEL_DATE", "2018-06-02"
)

PR_AUC = Gauge("ml_performance_pr_auc", "PR-AUC of labeled predictions")
ROC_AUC = Gauge("ml_performance_roc_auc", "ROC-AUC of labeled predictions")
PRECISION = Gauge("ml_performance_precision", "Precision")
RECALL = Gauge("ml_performance_recall", "Recall")
COVERAGE = Gauge("ml_performance_label_coverage", "Fraction of predictions with available labels")
LABELED_COUNT = Gauge("ml_performance_labeled_count", "Number of labeled predictions")
POSITIVE_COUNT = Gauge("ml_performance_positive_count", "Number of positive labels")


def calculate_performance(events, labels, as_of_date, threshold=0.4335175879396985):
    if events.empty:
        return None

    required = {"order_id", "probability"}
    if not required.issubset(events.columns):
        raise ValueError("Prediction events missing required fields")

    cutoff = pd.Timestamp(as_of_date)
    labels = labels.copy()

    labels["review_creation_date"] = pd.to_datetime(
        labels["review_creation_date"]
    )

    # A date-only cutoff includes the full selected day.
    available = labels[
        labels["review_creation_date"] < cutoff + pd.Timedelta(days=1)
    ]

    if available["order_id"].duplicated().any():
        raise ValueError("Duplicate order IDs in labels")

    predictions = events.drop_duplicates(
        "order_id", keep="last"
    ).copy()

    joined = predictions.merge(
        available[["order_id", "is_bad_review"]].rename(
            columns={"is_bad_review": "true_label"}
        ),
        on="order_id",
        how="inner",
        validate="one_to_one",
    )

    total = len(predictions)
    labeled = len(joined)

    result = {
        "coverage": labeled / total,
        "labeled_count": labeled,
        "positive_count": 0,
        "pr_auc": float("nan"),
        "roc_auc": float("nan"),
        "precision": float("nan"),
        "recall": float("nan"),
    }

    if labeled == 0:
        return result

    y_true = joined["true_label"].astype(int)
    probabilities = joined["probability"].astype(float)

    result["positive_count"] = int(y_true.sum())

    y_pred = (probabilities >= threshold).astype(int)

    result["precision"] = float(
        precision_score(y_true, y_pred, zero_division=0)
    )
    result["recall"] = float(
        recall_score(y_true, y_pred, zero_division=0)
    )

    if y_true.nunique() == 2:
        result["pr_auc"] = float(
            average_precision_score(y_true, probabilities)
        )
        result["roc_auc"] = float(
            roc_auc_score(y_true, probabilities)
        )

    return result


def update_performance_metrics(events):
    labels = pd.read_parquet(LABELS_PATH)

    result = calculate_performance(
        events, labels, SIMULATED_DATE
    )

    if result is None:
        return

    COVERAGE.set(result["coverage"])
    LABELED_COUNT.set(result["labeled_count"])
    POSITIVE_COUNT.set(result["positive_count"])
    PR_AUC.set(result["pr_auc"])
    ROC_AUC.set(result["roc_auc"])
    PRECISION.set(result["precision"])
    RECALL.set(result["recall"])

    logger.info("Performance: %s", result)
