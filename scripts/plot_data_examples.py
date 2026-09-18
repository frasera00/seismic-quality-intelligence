"""Plot example seismic traces from the dataset and after classification."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from joblib import load

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
    parser.add_argument(
        "--n-examples",
        type=int,
        default=4,
        help="Number of example traces to show (2x2 grid).",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def find_example_indices(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_examples: int = 4,
    rng: np.random.Generator | None = None,
) -> dict[str, list[int]]:
    """Find example indices for TN, TP, FN, FP categories.

    Returns a dict with lists of indices for each category.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    tn_mask = (y_true == 0) & (y_pred == 0)
    tp_mask = (y_true == 1) & (y_pred == 1)
    fn_mask = (y_true == 1) & (y_pred == 0)
    fp_mask = (y_true == 0) & (y_pred == 1)

    categories = {
        "TN": tn_mask,
        "TP": tp_mask,
        "FN": fn_mask,
        "FP": fp_mask,
    }

    examples = {key: [] for key in categories}

    for key, mask in categories.items():
        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue
        # sample up to n_examples per category if desired; here we just take one
        chosen = rng.choice(idx, size=min(1, len(idx)), replace=False)
        examples[key].extend(chosen.tolist())

    return examples


def plot_trace(
    ax: plt.Axes,
    t: np.ndarray,
    trace: np.ndarray,
    title: str,
    color: str = "k",
) -> None:
    """Plot a single seismic trace on a given axes."""
    ax.plot(t, trace, color=color, linewidth=1)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


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
    time = data["time"]

    X = extract_features(traces, dt)
    y = labels

    # Train/validation split (same as in train.py)
    n_train = int(0.7 * len(X))
    rng = np.random.default_rng(data_cfg["seed"] + 1)
    perm = rng.permutation(len(X))
    val_idx = perm[n_train:]

    X_val = X[val_idx]
    y_val = y[val_idx]
    traces_val = traces[val_idx]

    # Load model and predict
    model = load(args.model_path)
    scores_val = model.decision_function(X_val)
    y_pred_val = (scores_val < 0).astype(np.int32)

    # Find example indices
    examples = find_example_indices(y_val, y_pred_val, rng=rng)

    # Build a 2x2 panel: TN, TP, FN, FP
    categories_order = ["TN", "TP", "FN", "FP"]
    titles = {
        "TN": "True Negative (clean, predicted normal)",
        "TP": "True Positive (anomaly, predicted anomaly)",
        "FN": "False Negative (anomaly, predicted normal)",
        "FP": "False Positive (clean, predicted anomaly)",
    }
    colors = {
        "TN": "C0",  # normal
        "TP": "C1",  # anomaly
        "FN": "C2",  # missed anomaly
        "FP": "C3",  # false alarm
    }

    fig, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(9, 6),
        sharex=True,
        sharey=False,
    )
    axes = axes.ravel()

    for i, key in enumerate(categories_order):
        ax = axes[i]
        if not examples[key]:
            ax.text(
                0.5,
                0.5,
                f"No {key} examples in validation set",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(titles[key], fontsize=10)
            continue

        idx = examples[key][0]
        trace = traces_val[idx]
        plot_trace(ax, time, trace, titles[key], color=colors[key])

    fig.suptitle(
        "Example seismic traces from validation set\n"
        "Top: clean vs anomalous; Bottom: classification outcome",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "data_examples.png", dpi=300)
    plt.close(fig)

    print(f"Data examples saved to {output_dir / 'data_examples.png'}")


if __name__ == "__main__":
    main()
