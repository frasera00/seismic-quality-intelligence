"""Feature extraction for trace-level QC within seismic gathers."""

from __future__ import annotations

import numpy as np
from scipy.fft import rfft, rfftfreq

FEATURE_NAMES = [
    "rms_amplitude",
    "peak_to_peak",
    "kurtosis",
    "dominant_frequency_hz",
    "spectral_entropy",
    "neighbor_correlation",
    "neighbor_residual_rms",
]


def _validate_gather(gather: np.ndarray) -> np.ndarray:
    """Validate and return a floating-point gather."""
    gather = np.asarray(gather, dtype=float)

    if gather.ndim != 2:
        raise ValueError("gather must have shape (n_traces, n_samples)")

    if gather.shape[0] < 3:
        raise ValueError("gather must contain at least 3 traces")

    if gather.shape[1] < 4:
        raise ValueError("gather must contain at least 4 samples")

    return gather


def trace_rms(trace: np.ndarray) -> float:
    """Return root-mean-square amplitude."""
    return float(np.sqrt(np.mean(trace**2)))


def trace_peak_to_peak(trace: np.ndarray) -> float:
    """Return peak-to-peak amplitude."""
    return float(np.ptp(trace))


def trace_kurtosis(trace: np.ndarray) -> float:
    """Return non-excess kurtosis of a trace amplitude distribution."""
    centered = trace - np.mean(trace)
    std = np.std(centered)

    if std == 0.0:
        return 0.0

    return float(np.mean(centered**4) / std**4)


def trace_dominant_frequency(
    trace: np.ndarray,
    dt: float,
) -> float:
    """Return dominant frequency from the amplitude spectrum."""
    spectrum = np.abs(rfft(trace))
    frequencies = rfftfreq(len(trace), dt)

    if np.allclose(spectrum, 0.0):
        return 0.0

    return float(frequencies[np.argmax(spectrum)])


def trace_spectral_entropy(
    trace: np.ndarray,
    eps: float = 1e-12,
) -> float:
    """Return normalized entropy of the trace power spectrum."""
    power = np.abs(rfft(trace)) ** 2
    power_sum = np.sum(power)

    if power_sum <= eps:
        return 0.0

    probabilities = power / power_sum
    probabilities = probabilities[probabilities > eps]

    entropy = -np.sum(probabilities * np.log(probabilities))
    maximum_entropy = np.log(len(probabilities))

    if maximum_entropy <= eps:
        return 0.0

    return float(entropy / maximum_entropy)


def neighbor_reference(
    gather: np.ndarray,
    trace_index: int,
) -> np.ndarray:
    """Build a robust local reference trace from immediate neighbors."""
    n_traces = gather.shape[0]

    if trace_index == 0:
        return gather[1]

    if trace_index == n_traces - 1:
        return gather[-2]

    return np.median(
        gather[trace_index - 1 : trace_index + 2 : 2],
        axis=0,
    )


def neighbor_correlation(
    trace: np.ndarray,
    reference: np.ndarray,
) -> float:
    """Return Pearson correlation with a neighboring-trace reference."""
    trace_std = np.std(trace)
    reference_std = np.std(reference)

    if trace_std == 0.0 or reference_std == 0.0:
        return 0.0

    return float(np.corrcoef(trace, reference)[0, 1])


def neighbor_residual_rms(
    trace: np.ndarray,
    reference: np.ndarray,
) -> float:
    """Return RMS amplitude of the residual relative to neighbors."""
    return trace_rms(trace - reference)


def extract_gather_features(
    gather: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Extract trace-level features from a seismic gather.

    Parameters
    ----------
    gather : np.ndarray
        Array with shape ``(n_traces, n_samples)``.
    dt : float
        Sampling interval in seconds.

    Returns
    -------
    np.ndarray
        Feature matrix with shape ``(n_traces, 7)``. The column order
        is given by ``FEATURE_NAMES``.
    """
    gather = _validate_gather(gather)

    if dt <= 0.0:
        raise ValueError("dt must be positive")

    feature_rows = []

    for trace_index, trace in enumerate(gather):
        reference = neighbor_reference(gather, trace_index)

        feature_rows.append(
            [
                trace_rms(trace),
                trace_peak_to_peak(trace),
                trace_kurtosis(trace),
                trace_dominant_frequency(trace, dt),
                trace_spectral_entropy(trace),
                neighbor_correlation(trace, reference),
                neighbor_residual_rms(trace, reference),
            ]
        )

    return np.asarray(feature_rows, dtype=float)


def flatten_gather_dataset(
    gathers: np.ndarray,
    labels: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert a gather dataset into trace features, labels, and group IDs.

    Parameters
    ----------
    gathers : np.ndarray
        Array with shape ``(n_gathers, n_traces, n_samples)``.
    labels : np.ndarray
        Binary labels with shape ``(n_gathers, n_traces)``.
    dt : float
        Sampling interval in seconds.

    Returns
    -------
    X : np.ndarray
        Trace-level feature matrix.
    y : np.ndarray
        Trace-level labels, 0 normal and 1 anomalous.
    groups : np.ndarray
        Gather identifier for each trace. Use this for group-aware splitting.
    """
    gathers = np.asarray(gathers, dtype=float)
    labels = np.asarray(labels, dtype=np.int32)

    if gathers.ndim != 3:
        raise ValueError("gathers must have shape (n_gathers, n_traces, n_samples)")

    if labels.shape != gathers.shape[:2]:
        raise ValueError("labels must have shape (n_gathers, n_traces)")

    feature_blocks = []
    label_blocks = []
    group_blocks = []

    for gather_index, gather in enumerate(gathers):
        features = extract_gather_features(
            gather=gather,
            dt=dt,
        )

        feature_blocks.append(features)
        label_blocks.append(labels[gather_index])
        group_blocks.append(
            np.full(
                gather.shape[0],
                gather_index,
                dtype=np.int32,
            )
        )

    return (
        np.vstack(feature_blocks),
        np.concatenate(label_blocks),
        np.concatenate(group_blocks),
    )
