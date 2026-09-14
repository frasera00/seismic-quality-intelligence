"""Evaluation metrics for anomaly detection."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    auc,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute standard classification metrics.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth binary labels (0 = normal, 1 = anomaly).
    y_pred : np.ndarray
        Predicted binary labels.

    Returns
    -------
    metrics : dict
        Dictionary with:
        - precision
        - recall
        - f1
        - tn, fp, fn, tp
    """
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    precision = precision_score(y_true, y_pred, zero_division=0.0)
    recall = recall_score(y_true, y_pred, zero_division=0.0)

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def compute_roc_auc(
    y_true: np.ndarray,
    scores: np.ndarray,
) -> float:
    """Compute ROC AUC for anomaly scores.

    Higher scores should correspond to more anomalous samples.
    Here we assume more negative scores are more anomalous, so we negate.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth binary labels.
    scores : np.ndarray
        Anomaly scores from the model.

    Returns
    -------
    roc_auc : float
        Area under the ROC curve.
    """
    if y_true.shape != scores.shape:
        raise ValueError("y_true and scores must have the same shape")

    fpr, tpr, _ = roc_curve(y_true, -scores)
    return float(auc(fpr, tpr))


def compute_pr_auc(
    y_true: np.ndarray,
    scores: np.ndarray,
) -> float:
    """Compute Precision-Recall AUC for anomaly scores.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth binary labels.
    scores : np.ndarray
        Anomaly scores from the model.

    Returns
    -------
    pr_auc : float
        Area under the Precision-Recall curve.
    """
    if y_true.shape != scores.shape:
        raise ValueError("y_true and scores must have the same shape")

    precision, recall, _ = precision_recall_curve(y_true, -scores)
    return float(auc(recall, precision))