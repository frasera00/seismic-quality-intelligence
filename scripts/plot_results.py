"""Plot evaluation results for the seismic QC model."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from joblib import load
from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
)

from seismic_quality_intelligence.features import extract_features
from seismic_quality_intelligence.synthetic import generate_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/baseline.yaml"),
        help="Path to the YAML experiment configuration.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("reports/model.joblib"),
        help="Path to the trained model.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/figures"),
        help="Directory to save figures.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    data_cfg = config["data"]
    dt = data_cfg["dt"]

    # Regenerate dataset with same seed
    data = generate_dataset(
        n_traces=data_cfg["n_traces"],
        anomaly_fraction=data_cfg["anomaly_fraction"],
        duration=data_cfg["duration"],
        dt=data_cfg["dt"],
        f0=data_cfg["f0"],
        seed=data_cfg["seed"],
    )

    traces = data["traces"]
    labels = data["labels"]

    X = extract_features(traces, dt)
    y = labels

    # Train/validation split (same as in train.py)
    n_train = int(0.7 * len(X))
    rng = np.random.default_rng(data_cfg["seed"] + 1)
    perm = rng.permutation(len(X))
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]

    X_val = X[val_idx]
    y_val = y[val_idx]

    # Load model
    model = load(args.model_path)

    scores_val = model.decision_function(X_val)

    # Prepare output directory
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # ROC curve
    fig_roc, ax_roc = plt.subplots(figsize=(5, 5))
    RocCurveDisplay.from_predictions(
        y_val,
        -scores_val,
        ax=ax_roc,
        plot_chance_level=True,
    )
    ax_roc.set_title("ROC curve (validation set)")
    fig_roc.tight_layout()
    fig_roc.savefig(output_dir / "roc_curve.png", dpi=300)
    plt.close(fig_roc)

    # PR curve
    fig_pr, ax_pr = plt.subplots(figsize=(5, 5))
    PrecisionRecallDisplay.from_predictions(
        y_val,
        -scores_val,
        ax=ax_pr,
    )
    ax_pr.set_title("Precision-Recall curve (validation set)")
    fig_pr.tight_layout()
    fig_pr.savefig(output_dir / "pr_curve.png", dpi=300)
    plt.close(fig_pr)

    # Score distributions
    fig_scores, ax_scores = plt.subplots(figsize=(6, 4))
    scores_normal = scores_val[y_val == 0]
    scores_anom = scores_val[y_val == 1]

    ax_scores.hist(
        scores_normal,
        bins=30,
        alpha=0.7,
        label="Normal",
        density=True,
    )
    ax_scores.hist(
        scores_anom,
        bins=30,
        alpha=0.7,
        label="Anomaly",
        density=True,
    )
    ax_scores.axvline(0.0, color="k", linestyle="--", linewidth=1)
    ax_scores.set_xlabel("Anomaly score (decision function)")
    ax_scores.set_ylabel("Density")
    ax_scores.legend()
    ax_scores.set_title("Distribution of anomaly scores")
    fig_scores.tight_layout()
    fig_scores.savefig(output_dir / "score_distribution.png", dpi=300)
    plt.close(fig_scores)

    print(f"Figures saved to {output_dir}")


if __name__ == "__main__":
    main()