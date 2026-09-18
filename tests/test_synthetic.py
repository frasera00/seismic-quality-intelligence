import numpy as np

from seismic_quality_intelligence.synthetic import (
    generate_clean_trace,
    generate_dataset,
    inject_anomaly,
    ricker_wavelet,
)


def test_ricker_wavelet_shape_and_peak() -> None:
    t, w = ricker_wavelet(duration=2.0, dt=0.002, f0=25.0, t0=1.0)

    assert t.ndim == 1
    assert w.ndim == 1
    assert len(t) == len(w)
    assert np.isclose(t[0], 0.0, atol=1e-6)
    assert np.argmax(np.abs(w)) == len(w) // 2


def test_generate_clean_trace_returns_valid_trace() -> None:
    t, trace = generate_clean_trace(duration=2.0, dt=0.002, f0=25.0)

    assert t.shape == trace.shape
    assert trace.ndim == 1
    assert np.std(trace) > 0.0


def test_inject_anomaly_dead_trace_zeros() -> None:
    rng = np.random.default_rng(0)
    _, trace = generate_clean_trace(
        duration=2.0,
        dt=0.002,
        f0=25.0,
        rng=rng,
    )

    anom, record = inject_anomaly(trace, "dead_trace", rng=rng)

    np.testing.assert_array_equal(anom, np.zeros_like(trace))


def test_generate_dataset_has_expected_shape_and_labels() -> None:
    data = generate_dataset(n_traces=200, anomaly_fraction=0.3, seed=42)

    traces = data["traces"]
    labels = data["labels"]
    anomaly_type = data["anomaly_type"]

    assert traces.ndim == 2
    assert traces.shape[0] == 200
    assert labels.shape == (200,)
    assert anomaly_type.shape == (200,)
    assert set(np.unique(labels)) == {0, 1}
    assert (labels == 1).sum() > 0
    assert (labels == 0).sum() > 0
