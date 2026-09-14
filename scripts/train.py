"""Train trace-level seismic QC anomaly detection on synthetic gathers."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml
from joblib import dump
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from seismic_quality_intelligence.detectors import train_isolation_forest
from seismic_quality_intelligence.evaluation import (
    compute_metrics,
    compute_pr_auc,
    compute_roc_auc,
)
from seismic_quality_intelligence.gather_features import (
    FEATURE_NAMES,
    flatten_gather_dataset,
)
from seismic_quality_intelligence.synthetic_gather import (
    generate_gather_dataset,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()

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
        help="Directory in which to store model and metrics outputs.",
    )

    return parser.parse_args()


def load_config(path: Path) -> dict:
    """Load YAML configuration."""
    with path.open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def save_yaml(data: dict, path: Path) -> None:
    """Save a dictionary to YAML with NumPy scalars converted to Python values."""
    serializable = {}

    for key, value in data.items():
        if isinstance(value, np.generic):
            serializable[key] = value.item()
        else:
            serializable[key] = value

    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(serializable, file, sort_keys=True)


def main() -> None:
    """Generate gathers, train an Isolation Forest, and evaluate by gather."""
    args = parse_args()
    config = load_config(args.config)

    data_cfg = config["data"]
    split_cfg = config["split"]
    model_cfg = config["model"]

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

    gathers = dataset["gathers"]
    labels = dataset["labels"]

    X, y, groups = flatten_gather_dataset(
        gathers=gathers,
        labels=labels,
        dt=data_cfg["dt"],
    )

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=split_cfg["validation_fraction"],
        random_state=split_cfg["random_state"],
    )

    train_idx, val_idx = next(
        splitter.split(
            X,
            y,
            groups=groups,
        )
    )

    X_train = X[train_idx]
    y_train = y[train_idx]
    X_val = X[val_idx]
    y_val = y[val_idx]

    normal_train_mask = y_train == 0

    scaler = StandardScaler()
    X_train_normal = scaler.fit_transform(X_train[normal_train_mask])
    X_val_scaled = scaler.transform(X_val)

    max_samples = model_cfg.get("max_samples", "auto")

    if max_samples == "auto":
        max_samples = "auto"

    detector = train_isolation_forest(
        X=X_train_normal,
        contamination=model_cfg["contamination"],
        random_state=model_cfg["random_state"],
        n_estimators=model_cfg["n_estimators"],
        max_samples=max_samples,
    )

    model = Pipeline(
        steps=[
            ("scaler", scaler),
            ("detector", detector),
        ]
    )

    scores_val = detector.decision_function(X_val_scaled)
    y_pred_val = (scores_val < 0.0).astype(np.int32)

    metrics = compute_metrics(
        y_true=y_val,
        y_pred=y_pred_val,
    )
    metrics["roc_auc"] = compute_roc_auc(
        y_true=y_val,
        scores=scores_val,
    )
    metrics["pr_auc"] = compute_pr_auc(
        y_true=y_val,
        scores=scores_val,
    )
    metrics["n_gathers_total"] = int(gathers.shape[0])
    metrics["n_gathers_train"] = int(len(np.unique(groups[train_idx])))
    metrics["n_gathers_val"] = int(len(np.unique(groups[val_idx])))
    metrics["n_traces_train"] = int(len(train_idx))
    metrics["n_traces_val"] = int(len(val_idx))
    metrics["n_normal_train"] = int(np.sum(normal_train_mask))
    metrics["n_anomalous_val"] = int(np.sum(y_val == 1))

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    dump(
        model,
        args.output_dir / "gather_qc_model.joblib",
    )

    save_yaml(
        metrics,
        args.output_dir / "gather_qc_metrics.yaml",
    )

    np.savez_compressed(
        args.output_dir / "validation_predictions.npz",
        y_true=y_val,
        y_pred=y_pred_val,
        scores=scores_val,
        val_indices=val_idx,
        groups=groups[val_idx],
    )

    print("\nSeismic Quality Intelligence — Gather-Level Baseline")
    print("-" * 56)
    print(f"Total gathers:      {metrics['n_gathers_total']}")
    print(
        f"Train/val gathers:  {metrics['n_gathers_train']} / {metrics['n_gathers_val']}"
    )
    print(
        f"Train/val traces:   {metrics['n_traces_train']} / {metrics['n_traces_val']}"
    )
    print(f"Normal train traces:{metrics['n_normal_train']}")
    print(f"Anomalous val traces:{metrics['n_anomalous_val']}")
    print(f"Features:           {len(FEATURE_NAMES)}")
    print("\nValidation metrics")

    for metric_name in [
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
    ]:
        print(f"  {metric_name:10s}: {metrics[metric_name]:.4f}")

    print("\nSaved outputs")
    print(f"  Model:   {args.output_dir / 'gather_qc_model.joblib'}")
    print(f"  Metrics: {args.output_dir / 'gather_qc_metrics.yaml'}")
    print(f"  Scores:  {args.output_dir / 'validation_predictions.npz'}")


if __name__ == "__main__":
    main()
