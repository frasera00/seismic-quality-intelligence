"""Synthetic seismic gathers with hyperbolic moveout and QC anomalies."""

from __future__ import annotations

from typing import Literal

import numpy as np


GatherAnomalyType = Literal[
    "spike",
    "dead_trace",
    "gain",
]


def ricker_wavelet(
    dt: float,
    f0: float,
    duration: float = 0.128,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a zero-phase Ricker wavelet centered at zero time.

    Parameters
    ----------
    dt : float
        Sampling interval in seconds.
    f0 : float
        Dominant frequency in Hz.
    duration : float
        Total wavelet duration in seconds.

    Returns
    -------
    time : np.ndarray
        Symmetric wavelet time axis in seconds.
    wavelet : np.ndarray
        Ricker wavelet samples.
    """
    if dt <= 0.0:
        raise ValueError("dt must be positive")

    if f0 <= 0.0:
        raise ValueError("f0 must be positive")

    if duration <= 0.0:
        raise ValueError("duration must be positive")

    half_duration = duration / 2.0
    time = np.arange(
        -half_duration,
        half_duration + dt,
        dt,
    )

    pi_f_t = np.pi * f0 * time
    wavelet = (1.0 - 2.0 * pi_f_t**2) * np.exp(-pi_f_t**2)

    return time, wavelet


def hyperbolic_time(
    offsets: np.ndarray,
    t0: float,
    velocity: float,
) -> np.ndarray:
    """Calculate two-way reflection time following hyperbolic moveout.

    Parameters
    ----------
    offsets : np.ndarray
        Source-receiver offsets in metres.
    t0 : float
        Zero-offset two-way travel time in seconds.
    velocity : float
        RMS velocity in metres per second.

    Returns
    -------
    np.ndarray
        Travel times for all offsets in seconds.
    """
    if t0 <= 0.0:
        raise ValueError("t0 must be positive")

    if velocity <= 0.0:
        raise ValueError("velocity must be positive")

    offsets = np.asarray(offsets, dtype=float)

    return np.sqrt(t0**2 + (offsets / velocity) ** 2)


def add_wavelet_at_time(
    trace: np.ndarray,
    wavelet: np.ndarray,
    sample_index: int,
    amplitude: float,
) -> None:
    """Add a wavelet to a trace in place, clipping it at trace boundaries."""
    half_length = len(wavelet) // 2

    start_trace = max(0, sample_index - half_length)
    end_trace = min(len(trace), sample_index + half_length + 1)

    start_wavelet = start_trace - (sample_index - half_length)
    end_wavelet = start_wavelet + (end_trace - start_trace)

    trace[start_trace:end_trace] += (
        amplitude * wavelet[start_wavelet:end_wavelet]
    )


def generate_clean_gather(
    n_traces: int = 96,
    duration: float = 2.5,
    dt: float = 0.002,
    max_offset: float = 2500.0,
    reflector_times: tuple[float, ...] = (0.45, 0.9, 1.4, 1.9),
    reflector_velocities: tuple[float, ...] = (
        1800.0,
        2200.0,
        2600.0,
        3000.0,
    ),
    reflector_amplitudes: tuple[float, ...] = (
        0.8,
        -0.6,
        0.7,
        -0.5,
    ),
    f0: float = 28.0,
    noise_std: float = 0.03,
    rng: np.random.Generator | None = None,
) -> dict[str, np.ndarray]:
    """Generate a synthetic seismic gather containing hyperbolic events.

    The gather contains one trace for each offset. Every reflector follows
    hyperbolic normal moveout, and each event is represented by a Ricker
    wavelet.

    Parameters
    ----------
    n_traces : int
        Number of traces across the gather.
    duration : float
        Recording length in seconds.
    dt : float
        Sampling interval in seconds.
    max_offset : float
        Maximum absolute source-receiver offset in metres.
    reflector_times : tuple[float, ...]
        Zero-offset reflection times in seconds.
    reflector_velocities : tuple[float, ...]
        RMS velocities in metres per second.
    reflector_amplitudes : tuple[float, ...]
        Reflection amplitudes.
    f0 : float
        Dominant wavelet frequency in Hz.
    noise_std : float
        Standard deviation of additive Gaussian noise.
    rng : np.random.Generator, optional
        Random generator.

    Returns
    -------
    dict[str, np.ndarray]
        ``gather`` has shape ``(n_traces, n_samples)``.
        ``offsets`` has shape ``(n_traces,)``.
        ``time`` has shape ``(n_samples,)``.
    """
    if n_traces < 2:
        raise ValueError("n_traces must be at least 2")

    if len(reflector_times) != len(reflector_velocities):
        raise ValueError("reflector times and velocities must have equal length")

    if len(reflector_times) != len(reflector_amplitudes):
        raise ValueError("reflector times and amplitudes must have equal length")

    if rng is None:
        rng = np.random.default_rng()

    time = np.arange(0.0, duration, dt)
    offsets = np.linspace(-max_offset, max_offset, n_traces)
    gather = np.zeros((n_traces, len(time)), dtype=float)

    _, wavelet = ricker_wavelet(
        dt=dt,
        f0=f0,
    )

    for t0, velocity, amplitude in zip(
        reflector_times,
        reflector_velocities,
        reflector_amplitudes,
        strict=True,
    ):
        moveout_times = hyperbolic_time(
            offsets=offsets,
            t0=t0,
            velocity=velocity,
        )

        for trace_index, reflection_time in enumerate(moveout_times):
            sample_index = int(np.round(reflection_time / dt))

            if 0 <= sample_index < len(time):
                add_wavelet_at_time(
                    trace=gather[trace_index],
                    wavelet=wavelet,
                    sample_index=sample_index,
                    amplitude=amplitude,
                )

    gather += rng.normal(
        loc=0.0,
        scale=noise_std,
        size=gather.shape,
    )

    return {
        "gather": gather,
        "offsets": offsets,
        "time": time,
    }


def inject_trace_anomaly(
    gather: np.ndarray,
    trace_index: int,
    anomaly_type: GatherAnomalyType,
    rng: np.random.Generator | None = None,
    **kwargs,
) -> dict[str, object]:
    """Inject a QC anomaly into one trace of a gather in place.

    Parameters
    ----------
    gather : np.ndarray
        Seismic data with shape ``(n_traces, n_samples)``.
    trace_index : int
        Trace to modify.
    anomaly_type : GatherAnomalyType
        ``spike``, ``dead_trace``, or ``gain``.
    rng : np.random.Generator, optional
        Random generator.
    **kwargs
        Anomaly-specific settings.

    Returns
    -------
    dict[str, object]
        Metadata describing the injected anomaly.
    """
    if gather.ndim != 2:
        raise ValueError("gather must have shape (n_traces, n_samples)")

    if not 0 <= trace_index < gather.shape[0]:
        raise ValueError("trace_index is outside the gather")

    if rng is None:
        rng = np.random.default_rng()

    trace = gather[trace_index]
    n_samples = trace.size

    metadata: dict[str, object] = {
        "trace_index": trace_index,
        "type": anomaly_type,
        "sample_indices": [],
        "parameters": {},
    }

    if anomaly_type == "spike":
        n_spikes = int(kwargs.get("n_spikes", 2))
        spike_scale = float(kwargs.get("spike_scale", 12.0))

        if n_spikes < 1:
            raise ValueError("n_spikes must be at least 1")

        available = np.arange(10, n_samples - 10)

        if n_spikes > available.size:
            raise ValueError("n_spikes exceeds available trace samples")

        spike_indices = rng.choice(
            available,
            size=n_spikes,
            replace=False,
        )

        scale = spike_scale * np.std(trace)

        for sample_index in spike_indices:
            polarity = rng.choice([-1.0, 1.0])
            trace[sample_index] += polarity * scale

        metadata["sample_indices"] = sorted(
            int(index) for index in spike_indices
        )
        metadata["parameters"] = {
            "n_spikes": n_spikes,
            "spike_scale": spike_scale,
        }

    elif anomaly_type == "dead_trace":
        trace[:] = 0.0

        metadata["sample_indices"] = list(range(n_samples))

    elif anomaly_type == "gain":
        gain = float(kwargs.get("gain", 4.0))
        trace *= gain

        metadata["sample_indices"] = list(range(n_samples))
        metadata["parameters"] = {
            "gain": gain,
        }

    else:
        raise ValueError(f"Unknown anomaly type: {anomaly_type}")

    return metadata


def generate_gather_dataset(
    n_gathers: int = 100,
    n_traces: int = 96,
    anomaly_fraction: float = 0.10,
    seed: int = 42,
    **gather_kwargs,
) -> dict[str, np.ndarray]:
    """Generate synthetic gathers with randomly injected trace-level anomalies.

    Each gather has its own clean hyperbolic events and a subset of anomalous
    traces. Labels are defined at trace level.

    Parameters
    ----------
    n_gathers : int
        Number of gathers.
    n_traces : int
        Number of traces in each gather.
    anomaly_fraction : float
        Fraction of traces per gather that contain anomalies.
    seed : int
        Random seed.
    **gather_kwargs
        Parameters passed to ``generate_clean_gather``.

    Returns
    -------
    dict[str, np.ndarray]
        - ``gathers``: ``(n_gathers, n_traces, n_samples)``
        - ``labels``: ``(n_gathers, n_traces)``, 0 normal / 1 anomalous
        - ``anomaly_type``: anomaly label per trace
        - ``offsets``: offset axis
        - ``time``: time axis
    """
    if n_gathers < 1:
        raise ValueError("n_gathers must be at least 1")

    if not 0.0 < anomaly_fraction < 1.0:
        raise ValueError("anomaly_fraction must be between 0 and 1")

    rng = np.random.default_rng(seed)

    n_anomalous = max(
        1,
        int(np.round(anomaly_fraction * n_traces)),
    )

    gathers = []
    labels = []
    anomaly_types = []
    anomaly_records = []

    for gather_index in range(n_gathers):
        clean = generate_clean_gather(
            n_traces=n_traces,
            rng=rng,
            **gather_kwargs,
        )

        gather = clean["gather"]
        gather_labels = np.zeros(n_traces, dtype=np.int32)
        gather_types = np.full(n_traces, "clean", dtype="<U16")

        selected_traces = rng.choice(
            n_traces,
            size=n_anomalous,
            replace=False,
        )

        records_for_gather = []

        for trace_index in selected_traces:
            anomaly_type = str(
                rng.choice(
                    [
                        "spike",
                        "dead_trace",
                        "gain",
                    ]
                )
            )

            metadata = inject_trace_anomaly(
                gather=gather,
                trace_index=int(trace_index),
                anomaly_type=anomaly_type,
                rng=rng,
            )

            gather_labels[trace_index] = 1
            gather_types[trace_index] = anomaly_type
            records_for_gather.append(metadata)

        gathers.append(gather)
        labels.append(gather_labels)
        anomaly_types.append(gather_types)
        anomaly_records.append(records_for_gather)

    return {
        "gathers": np.stack(gathers, axis=0),
        "labels": np.stack(labels, axis=0),
        "anomaly_type": np.stack(anomaly_types, axis=0),
        "offsets": clean["offsets"],
        "time": clean["time"],
        "anomaly_records": np.asarray(anomaly_records, dtype=object),
    }