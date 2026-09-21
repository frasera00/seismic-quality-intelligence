"""Visualize synthetic gathers and injected QC anomalies."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from seismic_quality_intelligence.synthetic_gather import (
    generate_gather_dataset,
)


def plot_gather(
    ax: plt.Axes,
    gather: np.ndarray,
    offsets: np.ndarray,
    time: np.ndarray,
    labels: np.ndarray,
    anomaly_types: np.ndarray,
    title: str,
) -> None:
    """Display a gather and mark anomalous trace positions."""
    amplitude_scale = np.percentile(np.abs(gather), 99)

    ax.imshow(
        gather.T,
        cmap="seismic",
        aspect="auto",
        vmin=-amplitude_scale,
        vmax=amplitude_scale,
        extent=[
            offsets.min(),
            offsets.max(),
            time.max(),
            time.min(),
        ],
    )

    anomaly_indices = np.where(labels == 1)[0]

    for trace_index in anomaly_indices:
        offset = offsets[trace_index]

        ax.axvline(
            offset,
            color="gold",
            linewidth=1.4,
            alpha=0.9,
        )

        ax.text(
            offset,
            time.max() - 0.3,
            str(anomaly_types[trace_index]),
            rotation=90,
            color="black",
            fontsize=7,
            ha="center",
            va="top",
            clip_on=True,
            bbox={
                "facecolor": "gold",
                "edgecolor": "none",
                "alpha": 0.85,
                "pad": 1.5,
            },
        )

    ax.set_title(title)
    ax.set_xlabel("Offset (m)")
    ax.set_ylabel("Two-way travel time (s)")


def main() -> None:
    """Generate a clean-vs-anomalous gather comparison figure."""
    output_dir = Path("reports/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = generate_gather_dataset(
        n_gathers=2,
        n_traces=96,
        anomaly_fraction=0.10,
        duration=2.5,
        dt=0.002,
        max_offset=2500.0,
        seed=42,
    )

    gathers = dataset["gathers"]
    labels = dataset["labels"]
    anomaly_types = dataset["anomaly_type"]
    offsets = dataset["offsets"]
    time = dataset["time"]

    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(15, 7),
        sharey=True,
    )

    plot_gather(
        ax=axes[0],
        gather=gathers[0],
        offsets=offsets,
        time=time,
        labels=labels[0],
        anomaly_types=anomaly_types[0],
        title="Synthetic gather with trace-level QC anomalies",
    )

    plot_gather(
        ax=axes[1],
        gather=gathers[1],
        offsets=offsets,
        time=time,
        labels=labels[1],
        anomaly_types=anomaly_types[1],
        title="Independent synthetic gather with QC anomalies",
    )

    fig.suptitle(
        "Synthetic hyperbolic seismic gathers for QC anomaly detection",
        fontsize=14,
    )

    fig.tight_layout()

    output_path = output_dir / "synthetic_gather_examples.png"
    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    print(f"Saved figure: {output_path}")


if __name__ == "__main__":
    main()
