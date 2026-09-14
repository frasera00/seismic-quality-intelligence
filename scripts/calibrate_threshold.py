"""Calibrate Isolation Forest operating thresholds on seismic gathers."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from seismic_quality_intelligence.detectors import train_isolation_forest
from seismic_quality_intelligence.gather_features import (
    flatten_gather_dataset,
)
from seismic_quality_intelligence.synthetic_gather import (
    generate_gather_dataset,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Train an Isolation Forest on detector-training gathers, "
            "calibrate decision thresholds on separate calibration gathers, "
            "and evaluate once on held-out validation gathers."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/baseline.yaml"),
        help="Path to the YAML experiment configuration.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports"),
        help="Directory for calibration artifacts.",
    )
    parser.add_argument(
        "--calibration-fraction",
        type=float,
        default=0.25,
        help=("Fraction of outer-training gathers reserved for threshold calibration."),
    )

    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    """Load a YAML configuration file."""
    with path.open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def write_yaml(data: dict[str, Any], path: Path) -> None:
    """Write a plain-Python dictionary to YAML."""
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(
            data,
            file,
            sort_keys=False,
        )


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float | int]:
    """Compute binary metrics and the confusion matrix."""
    y_true = np.asarray(y_true, dtype=np.int32)
    y_pred = np.asarray(y_pred, dtype=np.int32)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def candidate_thresholds(scores: np.ndarray) -> np.ndarray:
    """Return midpoints between sorted unique anomaly scores."""
    scores = np.asarray(scores, dtype=float)
    unique_scores = np.sort(np.unique(scores))

    if unique_scores.size < 2:
        raise ValueError("At least two distinct calibration scores are required.")

    return (unique_scores[:-1] + unique_scores[1:]) / 2.0


def find_f1_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
) -> tuple[float, dict[str, float | int]]:
    """Choose the calibration threshold with the highest F1 score."""
    best_threshold: float | None = None
    best_metrics: dict[str, float | int] | None = None
    best_f1 = -np.inf

    for threshold in candidate_thresholds(scores):
        y_pred = (scores < threshold).astype(np.int32)
        metrics = compute_metrics(y_true, y_pred)

        if float(metrics["f1"]) > best_f1:
            best_f1 = float(metrics["f1"])
            best_threshold = float(threshold)
            best_metrics = metrics

    if best_threshold is None or best_metrics is None:
        raise RuntimeError("Unable to find an F1-optimal threshold.")

    return best_threshold, best_metrics


def find_precision_at_recall_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    recall_target: float,
) -> tuple[float, dict[str, float | int]]:
    """Maximize precision subject to an overall recall constraint."""
    best_threshold: float | None = None
    best_metrics: dict[str, float | int] | None = None
    best_precision = -np.inf

    for threshold in candidate_thresholds(scores):
        y_pred = (scores < threshold).astype(np.int32)
        metrics = compute_metrics(y_true, y_pred)

        if float(metrics["recall"]) < recall_target:
            continue

        if float(metrics["precision"]) > best_precision:
            best_precision = float(metrics["precision"])
            best_threshold = float(threshold)
            best_metrics = metrics

    if best_threshold is None or best_metrics is None:
        raise RuntimeError(
            "No threshold satisfied the requested overall recall target."
        )

    return best_threshold, best_metrics


def find_gain_safe_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    anomaly_types: np.ndarray,
    overall_recall_target: float,
    gain_recall_target: float,
) -> tuple[float, dict[str, float | int]]:
    """Maximize precision while protecting overall and gain recall.

    A trace is anomalous if its score is below the selected threshold.
    The selected threshold must satisfy both:

    - Overall anomaly recall >= ``overall_recall_target``.
    - Recall on anomalies labelled ``gain`` >= ``gain_recall_target``.
    """
    y_true = np.asarray(y_true, dtype=np.int32)
    scores = np.asarray(scores, dtype=float)
    anomaly_types = np.asarray(anomaly_types, dtype=str)

    gain_mask = (y_true == 1) & (anomaly_types == "gain")

    if not np.any(gain_mask):
        raise ValueError(
            "No labelled gain anomalies exist in the calibration partition."
        )

    gain_count = int(np.sum(gain_mask))

    best_threshold: float | None = None
    best_metrics: dict[str, float | int] | None = None
    best_precision = -np.inf

    for threshold in candidate_thresholds(scores):
        y_pred = (scores < threshold).astype(np.int32)
        metrics = compute_metrics(y_true, y_pred)

        gain_tp = int(np.sum(y_pred[gain_mask] == 1))
        gain_recall = gain_tp / gain_count

        if float(metrics["recall"]) < overall_recall_target:
            continue

        if gain_recall < gain_recall_target:
            continue

        if float(metrics["precision"]) > best_precision:
            best_precision = float(metrics["precision"])
            best_threshold = float(threshold)
            best_metrics = {
                **metrics,
                "gain_recall": float(gain_recall),
                "gain_tp": gain_tp,
                "gain_fn": gain_count - gain_tp,
            }

    if best_threshold is None or best_metrics is None:
        raise RuntimeError(
            "No threshold satisfied both the overall and gain-recall "
            "constraints. Lower one or both targets, or improve gain "
            "features/modeling."
        )

    return best_threshold, best_metrics


def anomaly_type_recall(
    anomaly_types: np.ndarray,
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
) -> dict[str, dict[str, dict[str, float | int]]]:
    """Calculate recall, TP, and FN separately for every anomaly type."""
    anomaly_types = np.asarray(anomaly_types, dtype=str)
    y_true = np.asarray(y_true, dtype=np.int32)

    results: dict[str, dict[str, dict[str, float | int]]] = {}
    anomaly_mask = y_true == 1

    for anomaly_type in sorted(np.unique(anomaly_types[anomaly_mask])):
        type_mask = anomaly_mask & (anomaly_types == anomaly_type)
        n_anomalies = int(np.sum(type_mask))

        results[str(anomaly_type)] = {}

        for policy_name, y_pred in predictions.items():
            y_pred = np.asarray(y_pred, dtype=np.int32)

            tp = int(np.sum(y_pred[type_mask] == 1))
            fn = int(np.sum(y_pred[type_mask] == 0))
            recall = tp / n_anomalies if n_anomalies else 0.0

            results[str(anomaly_type)][policy_name] = {
                "n_anomalies": n_anomalies,
                "tp": tp,
                "fn": fn,
                "recall": float(recall),
            }

    return results


def print_metrics(
    name: str,
    threshold: float,
    metrics: dict[str, float | int],
) -> None:
    """Print one threshold policy's aggregate metrics."""
    print(f"\n  {name}")
    print(f"    Threshold: {threshold:.5f}")
    print(f"    Precision: {float(metrics['precision']):.4f}")
    print(f"    Recall:    {float(metrics['recall']):.4f}")
    print(f"    F1:        {float(metrics['f1']):.4f}")
    print(
        "    TP / FP / TN / FN: "
        f"{int(metrics['tp'])} / "
        f"{int(metrics['fp'])} / "
        f"{int(metrics['tn'])} / "
        f"{int(metrics['fn'])}"
    )

    if "gain_recall" in metrics:
        print(f"    Gain recall: {float(metrics['gain_recall']):.4f}")
        print(
            f"    Gain TP / FN: {int(metrics['gain_tp'])} / {int(metrics['gain_fn'])}"
        )


def print_anomaly_type_recall(
    title: str,
    type_metrics: dict[str, dict[str, dict[str, float | int]]],
) -> None:
    """Print per-anomaly-type recall for every threshold policy."""
    print(f"\n{title}")
    print("-" * len(title))

    for anomaly_type, policy_results in type_metrics.items():
        print(f"\n  {anomaly_type}")

        for policy_name, metrics in policy_results.items():
            print(
                f"    {policy_name:24s}"
                f"recall={float(metrics['recall']):.4f} "
                f"TP={int(metrics['tp'])} "
                f"FN={int(metrics['fn'])} "
                f"N={int(metrics['n_anomalies'])}"
            )


def main() -> None:
    """Run group-aware training, calibration, policy selection, and testing."""
    args = parse_args()

    if not 0.0 < args.calibration_fraction < 1.0:
        raise ValueError("--calibration-fraction must be between 0 and 1.")

    config = load_config(args.config)

    data_cfg = config["data"]
    split_cfg = config["split"]
    model_cfg = config["model"]
    policy_cfg = config["operating_policy"]

    policy_name = str(policy_cfg["name"])
    overall_recall_target = float(policy_cfg["overall_recall_target"])
    gain_recall_target = float(policy_cfg["gain_recall_target"])
    review_threshold = float(policy_cfg["review_threshold"])

    if not 0.0 < overall_recall_target <= 1.0:
        raise ValueError("operating_policy.overall_recall_target must be in (0, 1].")

    if not 0.0 < gain_recall_target <= 1.0:
        raise ValueError("operating_policy.gain_recall_target must be in (0, 1].")

    dataset = generate_gather_dataset(
        n_gathers=data_cfg["n_gathers"],
        n_traces=data_cfg["n_traces"],
        anomaly_fraction=data_cfg["anomaly_fraction"],
        duration=data_cfg["duration"],
        dt=data_cfg["dt"],
        max_offset=data_cfg["max_offset"],
        f0=data_cfg["f0"],
        noise_std=data_cfg["noise_std"],
        seed=data_cfg["seed"],
    )

    gathers = np.asarray(dataset["gathers"])
    labels = np.asarray(dataset["labels"], dtype=np.int32)
    anomaly_type_grid = np.asarray(
        dataset["anomaly_type"],
        dtype=str,
    )

    X, y, groups = flatten_gather_dataset(
        gathers=gathers,
        labels=labels,
        dt=data_cfg["dt"],
    )

    anomaly_types = anomaly_type_grid.reshape(-1)

    if anomaly_types.shape[0] != X.shape[0]:
        raise ValueError(
            "The flattened anomaly-type grid does not match X. "
            "Expected anomaly_type shape (n_gathers, n_traces)."
        )

    outer_splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=split_cfg["validation_fraction"],
        random_state=split_cfg["random_state"],
    )

    train_idx, val_idx = next(
        outer_splitter.split(
            X,
            y,
            groups=groups,
        )
    )

    X_train = X[train_idx]
    y_train = y[train_idx]
    groups_train = groups[train_idx]
    types_train = anomaly_types[train_idx]

    X_val = X[val_idx]
    y_val = y[val_idx]
    groups_val = groups[val_idx]
    types_val = anomaly_types[val_idx]

    calibration_splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=args.calibration_fraction,
        random_state=split_cfg["random_state"] + 1,
    )

    detector_train_idx, calibration_idx = next(
        calibration_splitter.split(
            X_train,
            y_train,
            groups=groups_train,
        )
    )

    X_detector_train = X_train[detector_train_idx]
    y_detector_train = y_train[detector_train_idx]

    X_calib = X_train[calibration_idx]
    y_calib = y_train[calibration_idx]
    types_calib = types_train[calibration_idx]

    normal_detector_mask = y_detector_train == 0

    if not np.any(normal_detector_mask):
        raise ValueError("No normal traces are available for detector training.")

    scaler = StandardScaler()

    X_detector_train_normal = scaler.fit_transform(
        X_detector_train[normal_detector_mask]
    )

    X_calib_scaled = scaler.transform(X_calib)
    X_val_scaled = scaler.transform(X_val)

    detector = train_isolation_forest(
        X=X_detector_train_normal,
        contamination=model_cfg["contamination"],
        random_state=model_cfg["random_state"],
        n_estimators=model_cfg["n_estimators"],
        max_samples=model_cfg.get("max_samples", "auto"),
    )

    scores_calib = detector.decision_function(X_calib_scaled)
    scores_val = detector.decision_function(X_val_scaled)

    threshold_f1, calib_metrics_f1 = find_f1_threshold(
        y_true=y_calib,
        scores=scores_calib,
    )

    threshold_precision_recall, calib_metrics_precision_recall = (
        find_precision_at_recall_threshold(
            y_true=y_calib,
            scores=scores_calib,
            recall_target=overall_recall_target,
        )
    )

    threshold_gain_safe, calib_metrics_gain_safe = find_gain_safe_threshold(
        y_true=y_calib,
        scores=scores_calib,
        anomaly_types=types_calib,
        overall_recall_target=overall_recall_target,
        gain_recall_target=gain_recall_target,
    )

    predictions_calib = {
        "default": (scores_calib < 0.0).astype(np.int32),
        "f1_optimal": (scores_calib < threshold_f1).astype(np.int32),
        "precision_at_recall": (scores_calib < threshold_precision_recall).astype(
            np.int32
        ),
        "gain_safe_precision": (scores_calib < threshold_gain_safe).astype(np.int32),
    }

    predictions_val = {
        "default": (scores_val < 0.0).astype(np.int32),
        "f1_optimal": (scores_val < threshold_f1).astype(np.int32),
        "precision_at_recall": (scores_val < threshold_precision_recall).astype(
            np.int32
        ),
        "gain_safe_precision": (scores_val < threshold_gain_safe).astype(np.int32),
    }

    validation_metrics = {
        "default": compute_metrics(
            y_val,
            predictions_val["default"],
        ),
        "f1_optimal": compute_metrics(
            y_val,
            predictions_val["f1_optimal"],
        ),
        "precision_at_recall": compute_metrics(
            y_val,
            predictions_val["precision_at_recall"],
        ),
        "gain_safe_precision": compute_metrics(
            y_val,
            predictions_val["gain_safe_precision"],
        ),
    }

    calibration_type_metrics = anomaly_type_recall(
        anomaly_types=types_calib,
        y_true=y_calib,
        predictions=predictions_calib,
    )

    validation_type_metrics = anomaly_type_recall(
        anomaly_types=types_val,
        y_true=y_val,
        predictions=predictions_val,
    )

    policy_options = {
        "f1_optimal": {
            "threshold": threshold_f1,
            "calibration_metrics": calib_metrics_f1,
            "validation_metrics": validation_metrics["f1_optimal"],
        },
        "precision_at_recall": {
            "threshold": threshold_precision_recall,
            "calibration_metrics": (calib_metrics_precision_recall),
            "validation_metrics": validation_metrics["precision_at_recall"],
        },
        "gain_safe_precision": {
            "threshold": threshold_gain_safe,
            "calibration_metrics": calib_metrics_gain_safe,
            "validation_metrics": validation_metrics["gain_safe_precision"],
        },
    }

    if policy_name not in policy_options:
        available = ", ".join(sorted(policy_options))
        raise ValueError(
            f"Unknown operating policy '{policy_name}'. "
            f"Available policies: {available}."
        )

    selected_policy = policy_options[policy_name]
    selected_threshold = float(selected_policy["threshold"])
    selected_calibration_metrics = selected_policy["calibration_metrics"]
    selected_validation_metrics = selected_policy["validation_metrics"]

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    calibration_report = {
        "configuration": {
            "selected_policy": policy_name,
            "overall_recall_target": overall_recall_target,
            "gain_recall_target": gain_recall_target,
            "review_threshold": review_threshold,
            "calibration_fraction": args.calibration_fraction,
        },
        "calibration": {
            "n_gathers_outer_train": int(len(np.unique(groups_train))),
            "n_gathers_detector_train": int(
                len(np.unique(groups_train[detector_train_idx]))
            ),
            "n_gathers_calib": int(len(np.unique(groups_train[calibration_idx]))),
            "n_traces_calib": int(len(calibration_idx)),
            "n_anomalous_calib": int(np.sum(y_calib == 1)),
            "threshold_default": 0.0,
            "threshold_f1_optimal": float(threshold_f1),
            "threshold_precision_at_recall": float(threshold_precision_recall),
            "threshold_gain_safe_precision": float(threshold_gain_safe),
            "metrics_f1_optimal": calib_metrics_f1,
            "metrics_precision_at_recall": (calib_metrics_precision_recall),
            "metrics_gain_safe_precision": (calib_metrics_gain_safe),
            "anomaly_type_recall": calibration_type_metrics,
        },
        "validation": {
            "n_gathers_val": int(len(np.unique(groups_val))),
            "n_traces_val": int(len(val_idx)),
            "n_anomalous_val": int(np.sum(y_val == 1)),
            "metrics_default": validation_metrics["default"],
            "metrics_f1_optimal": validation_metrics["f1_optimal"],
            "metrics_precision_at_recall": validation_metrics["precision_at_recall"],
            "metrics_gain_safe_precision": validation_metrics["gain_safe_precision"],
            "anomaly_type_recall": validation_type_metrics,
        },
        "selected_operating_policy": {
            "policy_name": policy_name,
            "threshold": selected_threshold,
            "review_threshold": review_threshold,
            "calibration_metrics": selected_calibration_metrics,
            "validation_metrics": selected_validation_metrics,
        },
    }

    operating_policy = {
        "policy_name": policy_name,
        "threshold": selected_threshold,
        "review_threshold": review_threshold,
        "decision_rule": "flag when decision_score < threshold",
        "review_rule": (
            "manual review when threshold <= decision_score < review_threshold"
        ),
        "calibration_constraints": {
            "overall_recall_target": overall_recall_target,
            "gain_recall_target": gain_recall_target,
        },
        "calibration_metrics": selected_calibration_metrics,
        "validation_metrics": selected_validation_metrics,
        "validation_anomaly_type_recall": validation_type_metrics,
    }

    calibration_report_path = args.output_dir / "calibrated_thresholds.yaml"
    operating_policy_path = args.output_dir / "operating_policy.yaml"

    write_yaml(
        calibration_report,
        calibration_report_path,
    )
    write_yaml(
        operating_policy,
        operating_policy_path,
    )

    print("\nSeismic Quality Intelligence — Threshold Calibration")
    print("=" * 62)

    print("\nData partitions")
    print(
        "  Outer train / detector train / calibration gathers: "
        f"{len(np.unique(groups_train))} / "
        f"{len(np.unique(groups_train[detector_train_idx]))} / "
        f"{len(np.unique(groups_train[calibration_idx]))}"
    )
    print(
        "  Validation gathers / traces / anomalies: "
        f"{len(np.unique(groups_val))} / "
        f"{len(val_idx)} / "
        f"{int(np.sum(y_val == 1))}"
    )

    print("\nCalibration thresholds")
    print(f"  Default threshold:              {0.0:.5f}")
    print(f"  F1-optimal threshold:           {threshold_f1:.5f}")
    print(f"  Precision-at-recall threshold:  {threshold_precision_recall:.5f}")
    print(f"  Gain-safe precision threshold:  {threshold_gain_safe:.5f}")

    print("\nValidation metrics")
    print("-" * 62)
    print_metrics(
        "Default threshold",
        0.0,
        validation_metrics["default"],
    )
    print_metrics(
        "F1-optimal",
        threshold_f1,
        validation_metrics["f1_optimal"],
    )
    print_metrics(
        (f"Precision at recall >= {overall_recall_target:.2f}"),
        threshold_precision_recall,
        validation_metrics["precision_at_recall"],
    )
    print_metrics(
        (f"Gain-safe precision (gain recall >= {gain_recall_target:.2f})"),
        threshold_gain_safe,
        validation_metrics["gain_safe_precision"],
    )

    print_anomaly_type_recall(
        "Calibration recall by anomaly type",
        calibration_type_metrics,
    )
    print_anomaly_type_recall(
        "Validation recall by anomaly type",
        validation_type_metrics,
    )

    print("\nSelected operating policy")
    print(f"  Policy:           {policy_name}")
    print(f"  Alert threshold:  {selected_threshold:.5f}")
    print(f"  Review threshold: {review_threshold:.5f}")

    print("\nSaved artifacts")
    print(f"  Full report:      {calibration_report_path}")
    print(f"  Operating policy: {operating_policy_path}")


if __name__ == "__main__":
    main()
