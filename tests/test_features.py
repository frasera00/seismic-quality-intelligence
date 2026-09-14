import numpy as np
import pytest

from seismic_quality_intelligence.features import (
    dominant_frequency,
    energy_in_band,
    extract_features,
    kurtosis,
    mean_absolute_amplitude,
    peak_to_peak,
    rms_amplitude,
    skewness,
    spectral_entropy,
    zero_crossing_rate,
)


def test_rms_amplitude_basic() -> None:
    trace = np.array([3.0, 4.0])
    expected = np.sqrt((3.0**2 + 4.0**2) / 2.0)
    assert rms_amplitude(trace) == pytest.approx(expected)


def test_mean_absolute_amplitude_basic() -> None:
    trace = np.array([-2.0, 2.0])
    expected = 2.0
    assert mean_absolute_amplitude(trace) == pytest.approx(expected)


def test_peak_to_peak_basic() -> None:
    trace = np.array([-1.0, 0.0, 3.0])
    expected = 4.0
    assert peak_to_peak(trace) == pytest.approx(expected)


def test_zero_crossing_rate_simple() -> None:
    trace = np.array([-1.0, 1.0, -1.0, 1.0])
    dt = 0.002
    zcr = zero_crossing_rate(trace, dt)
    duration = len(trace) * dt
    assert zcr == pytest.approx(3.0 / duration)


def test_kurtosis_and_skewness_on_normal_data() -> None:
    rng = np.random.default_rng(0)
    trace = rng.normal(0.0, 1.0, size=1000)

    k = kurtosis(trace)
    s = skewness(trace)

    assert np.isfinite(k)
    assert np.isfinite(s)


def test_dominant_frequency_on_sinusoid() -> None:
    dt = 0.002
    f0 = 25.0
    t = np.arange(0.0, 2.0, dt)
    trace = np.sin(2 * np.pi * f0 * t)

    f_dom = dominant_frequency(trace, dt)
    assert np.isclose(f_dom, f0, rtol=0.05)


def test_spectral_entropy_non_negative() -> None:
    rng = np.random.default_rng(1)
    trace = rng.normal(0.0, 1.0, size=500)
    dt = 0.002

    entropy = spectral_entropy(trace, dt)
    assert 0.0 <= entropy <= 1.0


def test_energy_in_band_positive() -> None:
    rng = np.random.default_rng(2)
    trace = rng.normal(0.0, 1.0, size=500)
    dt = 0.002

    energy = energy_in_band(trace, dt, f_low=5.0, f_high=45.0)
    assert energy >= 0.0


def test_extract_features_shape() -> None:
    rng = np.random.default_rng(3)
    traces = rng.normal(0.0, 1.0, size=(10, 500))
    dt = 0.002

    X = extract_features(traces, dt)

    assert X.ndim == 2
    assert X.shape[0] == 10
    assert X.shape[1] == 9  # number of defined features