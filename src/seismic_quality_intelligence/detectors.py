"""Anomaly detectors for seismic trace quality control."""

from __future__ import annotations

from typing import Literal

import numpy as np
from sklearn.ensemble import IsolationForest

DetectorType = Literal["isolation_forest"]


def train_isolation_forest(
    X: np.ndarray,
    contamination: float = 0.1,
    random_state: int = 42,
    n_estimators: int = 200,
    max_samples: int | float | None = None,
) -> IsolationForest:
    """Train an Isolation Forest anomaly detector.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    contamination : float
        Expected proportion of anomalies in the training data.
    random_state : int
        Random seed for reproducibility.
    n_estimators : int
        Number of trees in the forest.
    max_samples : int or float, optional
        Number of samples to draw to train each tree.
        If None, uses 'auto' (sklearn default).

    Returns
    -------
    model : sklearn.ensemble.IsolationForest
        Trained Isolation Forest model.
    """
    max_samples_param = "auto" if max_samples is None else max_samples

    model = IsolationForest(
        n_estimators=n_estimators,
        max_samples=max_samples_param,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X)
    return model


def train_detector(
    X: np.ndarray,
    detector_type: DetectorType = "isolation_forest",
    **kwargs,
) -> object:
    """Factory function to train an anomaly detector.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    detector_type : DetectorType
        Type of detector. Currently supports:
        - "isolation_forest"
    **kwargs : dict
        Additional keyword arguments passed to the specific detector trainer.

    Returns
    -------
    model : object
        Trained detector model.
    """
    if detector_type == "isolation_forest":
        return train_isolation_forest(X, **kwargs)

    raise ValueError(f"Unknown detector type: {detector_type}")


def predict_anomaly_scores(
    model: IsolationForest,
    X: np.ndarray,
) -> np.ndarray:
    """Compute anomaly scores for samples.

    More negative scores indicate more anomalous samples.

    Parameters
    ----------
    model : IsolationForest
        Trained Isolation Forest model.
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).

    Returns
    -------
    scores : np.ndarray
        Anomaly scores (1D array).
    """
    return model.decision_function(X)


def predict_labels(
    model: IsolationForest,
    X: np.ndarray,
    threshold: float = 0.0,
) -> np.ndarray:
    """Predict binary anomaly labels from a trained model.

    Parameters
    ----------
    model : IsolationForest
        Trained Isolation Forest model.
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    threshold : float
        Decision threshold on the anomaly score.
        Samples with score < threshold are labeled as anomalies.

    Returns
    -------
    labels : np.ndarray
        Predicted labels: 0 = normal, 1 = anomaly.
    """
    scores = predict_anomaly_scores(model, X)
    labels = (scores < threshold).astype(np.int32)
    return labels
