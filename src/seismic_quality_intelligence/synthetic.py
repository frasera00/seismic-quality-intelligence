"""Synthetic seismic trace generation with controlled anomalies."""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy import signal as sig

AnomalyType = Literal[
    "noise_burst",
    "amplitude_gain",
    "polarity_flip",
    "dead_trace",
    "frequency_shift",
]


def ricker_wavelet(
    duration: float,
    dt: float,
    f0: float,
    t0: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a Ricker wavelet.

    Parameters
    ----------
    duration : float
        Total time duration in seconds.
    dt : float
        Time sample interval in seconds.
    f0 : float
        Dominant frequency in Hz.
    t0 : float, optional
        Time of the wavelet peak. Defaults to duration / 2.

    Returns
    -------
    time : np.ndarray
        Time axis in seconds.
    wavelet : np.ndarray
        Ricker wavelet amplitude.
    """
    if duration <= 0:
        raise ValueError("duration must be positive")

    if dt <= 0:
        raise ValueError("dt must be positive")

    if f0 <= 0:
        raise ValueError("f0 must be positive")

    if t0 is None:
        t0 = duration / 2.0

    time = np.arange(0.0, duration, dt)
    tau = time - t0
    a = (np.pi * f0 * tau) ** 2
    wavelet = (1.0 - 2.0 * a) * np.exp(-a)

    return time, wavelet


def generate_clean_trace(
    duration: float = 2.0,
    dt: float = 0.002,
    f0: float = 25.0,
    n_reflectors: int = 5,
    noise_std: float = 0.05,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic seismic trace without injected QC anomalies.

    The trace is a sum of shifted and scaled Ricker wavelets representing
    reflectors, plus additive Gaussian noise.

    Parameters
    ----------
    duration : float
        Total trace duration in seconds.
    dt : float
        Time sample interval in seconds.
    f0 : float
        Dominant frequency of the Ricker wavelet in Hz.
    n_reflectors : int
        Number of reflection events.
    noise_std : float
        Standard deviation of additive Gaussian noise.
    rng : np.random.Generator, optional
        Random generator for reproducible data generation.

    Returns
    -------
    time : np.ndarray
        Time axis in seconds.
    trace : np.ndarray
        Synthetic seismic trace.
    """
    if n_reflectors < 1:
        raise ValueError("n_reflectors must be at least 1")

    if noise_std < 0:
        raise ValueError("noise_std must be non-negative")

    if rng is None:
        rng = np.random.default_rng()

    time, wavelet = ricker_wavelet(
        duration=duration,
        dt=dt,
        f0=f0,
    )
    trace = np.zeros_like(time)

    reflection_times = rng.uniform(
        0.2 * duration,
        0.8 * duration,
        size=n_reflectors,
    )
    reflection_amplitudes = rng.uniform(
        -1.0,
        1.0,
        size=n_reflectors,
    )

    for reflection_time, amplitude in zip(
        reflection_times,
        reflection_amplitudes,
        strict=True,
    ):
        shift_samples = int(np.round(reflection_time / dt))

        if 0 < shift_samples < len(wavelet):
            trace[shift_samples:] += amplitude * wavelet[:-shift_samples]

    trace += rng.normal(
        loc=0.0,
        scale=noise_std,
        size=trace.shape,
    )

    return time, trace


def inject_anomaly(
    trace: np.ndarray,
    anomaly_type: AnomalyType,
    rng: np.random.Generator | None = None,
    **kwargs,
) -> tuple[np.ndarray, dict]:
    """Inject a controlled anomaly and return its metadata.

    The metadata records the anomaly type and sample interval. For localized
    anomalies, ``start`` and ``end`` identify the modified interval. For
    full-trace anomalies, the interval covers the full trace.

    Parameters
    ----------
    trace : np.ndarray
        One-dimensional input seismic trace.
    anomaly_type : AnomalyType
        Anomaly mechanism to inject.
    rng : np.random.Generator, optional
        Random generator for reproducible noise injection.
    **kwargs
        Optional anomaly-specific parameters.

    Returns
    -------
    trace_anom : np.ndarray
        Copy of the input trace with an anomaly injected.
    metadata : dict
        Metadata with ``type``, ``start``, ``end``, and ``parameters``.

    Raises
    ------
    ValueError
        If the trace is not one-dimensional, is empty, or the anomaly type
        is unsupported.
    """
    trace = np.asarray(trace, dtype=float)

    if trace.ndim != 1:
        raise ValueError("trace must be a one-dimensional array")

    if trace.size == 0:
        raise ValueError("trace must not be empty")

    if rng is None:
        rng = np.random.default_rng()

    trace_anom = trace.copy()
    n_samples = len(trace_anom)

    metadata = {
        "type": anomaly_type,
        "start": -1,
        "end": -1,
        "parameters": {},
    }

    if anomaly_type == "noise_burst":
        start = kwargs.get("start", int(0.4 * n_samples))
        length = kwargs.get("length", max(1, int(0.05 * n_samples)))
        scale = kwargs.get("scale", 5.0)

        if start < 0 or start >= n_samples:
            raise ValueError("noise_burst start must be within the trace")

        if length < 1:
            raise ValueError("noise_burst length must be at least 1")

        end = min(start + length, n_samples)
        noise_std = scale * np.std(trace)

        trace_anom[start:end] += rng.normal(
            loc=0.0,
            scale=noise_std,
            size=end - start,
        )

        metadata["start"] = start
        metadata["end"] = end
        metadata["parameters"] = {
            "scale": scale,
            "length": length,
        }

    elif anomaly_type == "amplitude_gain":
        start = kwargs.get("start", int(0.3 * n_samples))
        end = kwargs.get("end", int(0.7 * n_samples))
        gain = kwargs.get("gain", 3.0)

        if start < 0 or end > n_samples or start >= end:
            raise ValueError("amplitude_gain requires 0 <= start < end <= n_samples")

        trace_anom[start:end] *= gain

        metadata["start"] = start
        metadata["end"] = end
        metadata["parameters"] = {
            "gain": gain,
        }

    elif anomaly_type == "polarity_flip":
        start = kwargs.get("start", int(0.3 * n_samples))
        end = kwargs.get("end", int(0.7 * n_samples))

        if start < 0 or end > n_samples or start >= end:
            raise ValueError("polarity_flip requires 0 <= start < end <= n_samples")

        trace_anom[start:end] *= -1.0

        metadata["start"] = start
        metadata["end"] = end

    elif anomaly_type == "dead_trace":
        trace_anom[:] = 0.0

        metadata["start"] = 0
        metadata["end"] = n_samples

    elif anomaly_type == "frequency_shift":
        cutoff_ratio = kwargs.get("cutoff_ratio", 0.3)

        if not 0.0 < cutoff_ratio < 1.0:
            raise ValueError("cutoff_ratio must be between 0 and 1")

        b, a = sig.butter(
            N=4,
            Wn=cutoff_ratio,
            btype="low",
            output="ba",
        )
        trace_anom = sig.filtfilt(b, a, trace_anom)

        metadata["start"] = 0
        metadata["end"] = n_samples
        metadata["parameters"] = {
            "cutoff_ratio": cutoff_ratio,
        }

    else:
        raise ValueError(f"Unknown anomaly type: {anomaly_type}")

    return trace_anom, metadata


def generate_dataset(
    n_traces: int = 1000,
    anomaly_fraction: float = 0.3,
    duration: float = 2.0,
    dt: float = 0.002,
    f0: float = 25.0,
    anomaly_types: tuple[AnomalyType, ...] | None = None,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Generate a labeled dataset of synthetic seismic traces.

    Clean and anomalous traces are generated independently. Anomalous traces
    retain metadata identifying the anomaly mechanism and modified sample
    interval, allowing transparent visualization of the known synthetic ground
    truth.

    Parameters
    ----------
    n_traces : int
        Total number of traces to create.
    anomaly_fraction : float
        Fraction of traces that receive an injected anomaly.
    duration : float
        Trace duration in seconds.
    dt : float
        Time sample interval in seconds.
    f0 : float
        Dominant Ricker-wavelet frequency in Hz.
    anomaly_types : tuple[AnomalyType, ...], optional
        Anomaly types eligible for random selection. Uses all supported types
        by default.
    seed : int
        Random seed.

    Returns
    -------
    data : dict[str, np.ndarray]
        Dictionary containing:

        - ``traces``: array with shape ``(n_traces, n_samples)``
        - ``labels``: 0 for clean and 1 for anomalous traces
        - ``anomaly_type``: trace-level anomaly type
        - ``anomaly_start``: start sample of the injected anomaly, or -1
        - ``anomaly_end``: end sample of the injected anomaly, or -1
        - ``time``: common trace time axis in seconds.
    """
    if n_traces < 2:
        raise ValueError("n_traces must be at least 2")

    if not 0.0 < anomaly_fraction < 1.0:
        raise ValueError("anomaly_fraction must be between 0 and 1")

    if anomaly_types is None:
        anomaly_types = (
            "noise_burst",
            "amplitude_gain",
            "polarity_flip",
            "dead_trace",
            "frequency_shift",
        )

    if len(anomaly_types) == 0:
        raise ValueError("anomaly_types must contain at least one anomaly type")

    rng = np.random.default_rng(seed)

    n_anomalous = int(np.round(anomaly_fraction * n_traces))
    n_anomalous = max(1, min(n_anomalous, n_traces - 1))
    n_clean = n_traces - n_anomalous

    traces = []
    labels = []
    anomaly_type_list = []
    anomaly_start_list = []
    anomaly_end_list = []

    for _ in range(n_clean):
        _, trace = generate_clean_trace(
            duration=duration,
            dt=dt,
            f0=f0,
            rng=rng,
        )

        traces.append(trace)
        labels.append(0)
        anomaly_type_list.append("clean")
        anomaly_start_list.append(-1)
        anomaly_end_list.append(-1)

    for _ in range(n_anomalous):
        _, trace = generate_clean_trace(
            duration=duration,
            dt=dt,
            f0=f0,
            rng=rng,
        )

        anomaly_type = str(rng.choice(anomaly_types))

        trace_anom, metadata = inject_anomaly(
            trace=trace,
            anomaly_type=anomaly_type,
            rng=rng,
        )

        traces.append(trace_anom)
        labels.append(1)
        anomaly_type_list.append(anomaly_type)
        anomaly_start_list.append(metadata["start"])
        anomaly_end_list.append(metadata["end"])

    permutation = rng.permutation(n_traces)

    traces_arr = np.stack(traces, axis=0)[permutation]
    labels_arr = np.asarray(labels, dtype=np.int32)[permutation]
    anomaly_type_arr = np.asarray(anomaly_type_list, dtype=str)[permutation]
    anomaly_start_arr = np.asarray(
        anomaly_start_list,
        dtype=np.int32,
    )[permutation]
    anomaly_end_arr = np.asarray(
        anomaly_end_list,
        dtype=np.int32,
    )[permutation]

    time, _ = ricker_wavelet(
        duration=duration,
        dt=dt,
        f0=f0,
    )

    return {
        "traces": traces_arr,
        "labels": labels_arr,
        "anomaly_type": anomaly_type_arr,
        "anomaly_start": anomaly_start_arr,
        "anomaly_end": anomaly_end_arr,
        "time": time,
    }
