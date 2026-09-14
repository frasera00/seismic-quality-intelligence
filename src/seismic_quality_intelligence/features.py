"""Feature engineering for seismic traces."""

from __future__ import annotations

import numpy as np
from scipy import signal as sig
from scipy.fft import rfft, rfftfreq


def rms_amplitude(trace: np.ndarray) -> float:
    """Root-mean-square amplitude of a 1D trace."""
    trace = np.asarray(trace, dtype=float)
    if trace.ndim != 1:
        raise ValueError("trace must be a one-dimensional array")
    if trace.size == 0:
        raise ValueError("trace must not be empty")
    return float(np.sqrt(np.mean(trace**2)))


def mean_absolute_amplitude(trace: np.ndarray) -> float:
    """Mean absolute amplitude of a 1D trace."""
    trace = np.asarray(trace, dtype=float)
    if trace.ndim != 1:
        raise ValueError("trace must be a one-dimensional array")
    if trace.size == 0:
        raise ValueError("trace must not be empty")
    return float(np.mean(np.abs(trace)))


def peak_to_peak(trace: np.ndarray) -> float:
    """Peak-to-peak amplitude (max - min) of a 1D trace."""
    trace = np.asarray(trace, dtype=float)
    if trace.ndim != 1:
        raise ValueError("trace must be a one-dimensional array")
    if trace.size == 0:
        raise ValueError("trace must not be empty")
    return float(np.max(trace) - np.min(trace))


def zero_crossing_rate(trace: np.ndarray, dt: float) -> float:
    """Zero-crossing rate in crossings per second."""
    trace = np.asarray(trace, dtype=float)
    if trace.ndim != 1:
        raise ValueError("trace must be a one-dimensional array")
    if trace.size == 0:
        raise ValueError("trace must not be empty")

    signs = np.sign(trace)
    signs[signs == 0] = 1  # treat exact zeros as positive
    crossings = np.diff(signs) != 0
    duration = len(trace) * dt
    return float(np.sum(crossings) / duration)


def kurtosis(trace: np.ndarray) -> float:
    """Kurtosis of the trace amplitude distribution."""
    trace = np.asarray(trace, dtype=float)
    if trace.ndim != 1:
        raise ValueError("trace must be a one-dimensional array")
    if trace.size == 0:
        raise ValueError("trace must not be empty")
    n = len(trace)
    if n < 4:
        return np.nan
    mu = np.mean(trace)
    sigma = np.std(trace, ddof=0)
    if sigma == 0:
        return np.nan
    m4 = np.mean((trace - mu) ** 4)
    return float(m4 / (sigma**4))


def skewness(trace: np.ndarray) -> float:
    """Skewness of the trace amplitude distribution."""
    trace = np.asarray(trace, dtype=float)
    if trace.ndim != 1:
        raise ValueError("trace must be a one-dimensional array")
    if trace.size == 0:
        raise ValueError("trace must not be empty")
    n = len(trace)
    if n < 3:
        return np.nan
    mu = np.mean(trace)
    sigma = np.std(trace, ddof=0)
    if sigma == 0:
        return np.nan
    m3 = np.mean((trace - mu) ** 3)
    return float(m3 / (sigma**3))


def dominant_frequency(trace: np.ndarray, dt: float) -> float:
    """Dominant frequency (Hz) from the amplitude spectrum."""
    trace = np.asarray(trace, dtype=float)
    if trace.ndim != 1:
        raise ValueError("trace must be a one-dimensional array")
    if trace.size == 0:
        raise ValueError("trace must not be empty")

    n = len(trace)
    freqs = rfftfreq(n, dt)
    spectrum = np.abs(rfft(trace))

    if np.all(spectrum == 0):
        return np.nan

    idx = np.argmax(spectrum)
    return float(freqs[idx])


def spectral_entropy(trace: np.ndarray, dt: float, eps: float = 1e-12) -> float:
    """Spectral entropy of the trace (normalized)."""
    trace = np.asarray(trace, dtype=float)
    if trace.ndim != 1:
        raise ValueError("trace must be a one-dimensional array")
    if trace.size == 0:
        raise ValueError("trace must not be empty")

    n = len(trace)
    spectrum = np.abs(rfft(trace))
    power = spectrum**2
    total = np.sum(power)
    if total == 0:
        return 0.0
    p = power / (total + eps)
    p = p[p > 0]
    entropy = -np.sum(p * np.log(p))
    max_entropy = np.log(len(p))
    if max_entropy == 0:
        return 0.0
    return float(entropy / max_entropy)


def energy_in_band(
    trace: np.ndarray,
    dt: float,
    f_low: float,
    f_high: float,
) -> float:
    """Energy in a frequency band [f_low, f_high] (Hz)."""
    trace = np.asarray(trace, dtype=float)
    if trace.ndim != 1:
        raise ValueError("trace must be a one-dimensional array")
    if trace.size == 0:
        raise ValueError("trace must not be empty")

    n = len(trace)
    freqs = rfftfreq(n, dt)
    spectrum = np.abs(rfft(trace))
    power = spectrum**2

    mask = (freqs >= f_low) & (freqs <= f_high)
    return float(np.sum(power[mask]))


def extract_features(
    traces: np.ndarray,
    dt: float,
    f_low: float = 5.0,
    f_high: float = 45.0,
) -> np.ndarray:
    """Extract a feature vector for each trace in a batch.

    Parameters
    ----------
    traces : np.ndarray
        2D array of shape (n_traces, n_samples).
    dt : float
        Time sample interval in seconds.
    f_low : float
        Lower frequency for band energy.
    f_high : float
        Upper frequency for band energy.

    Returns
    -------
    X : np.ndarray
        2D array of shape (n_traces, n_features).
        Feature order:
        [rms, maa, p2p, zcr, kurtosis, skewness, dom_freq, spec_entropy, energy_band]
    """
    if traces.ndim != 2:
        raise ValueError("traces must be a 2D array")

    n_traces = traces.shape[0]

    features = []

    for i in range(n_traces):
        tr = traces[i]

        feat = [
            rms_amplitude(tr),
            mean_absolute_amplitude(tr),
            peak_to_peak(tr),
            zero_crossing_rate(tr, dt),
            kurtosis(tr),
            skewness(tr),
            dominant_frequency(tr, dt),
            spectral_entropy(tr, dt),
            energy_in_band(tr, dt, f_low, f_high),
        ]
        features.append(feat)

    return np.array(features, dtype=float)