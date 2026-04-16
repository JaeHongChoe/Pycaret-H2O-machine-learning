from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_score_column(predictions: pd.DataFrame) -> pd.Series | None:
    for column in ("Score", "prediction_score", "prediction_probability", "p1", "probability"):
        if column in predictions.columns:
            return predictions[column]
    return None


def classification_metrics(
    predictions: pd.DataFrame,
    *,
    truth_column: str,
    prediction_column: str = "prediction_label",
) -> dict[str, float | None]:
    """Compute a lightweight binary-classification metric bundle."""
    if truth_column not in predictions.columns or prediction_column not in predictions.columns:
        return {}

    y_true = predictions[truth_column]
    y_pred = predictions[prediction_column]
    y_score = _extract_score_column(predictions)

    metrics = {
        "accuracy": _to_float(accuracy_score(y_true, y_pred)),
        "precision": _to_float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": _to_float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": _to_float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": None,
    }

    if y_score is not None:
        try:
            metrics["auc"] = _to_float(roc_auc_score(y_true, y_score))
        except ValueError:
            metrics["auc"] = None

    return metrics
