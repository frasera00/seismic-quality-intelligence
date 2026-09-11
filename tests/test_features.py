import numpy as np
import pytest

from scientific_ml_template.features import rms_amplitude


def test_rms_amplitude_matches_expected_value() -> None:
    trace = np.array([3.0, 4.0])
    expected = np.sqrt((3.0**2 + 4.0**2) / 2.0)

    assert rms_amplitude(trace) == pytest.approx(expected)


def test_rms_amplitude_rejects_empty_trace() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        rms_amplitude(np.array([]))


def test_rms_amplitude_rejects_two_dimensional_input() -> None:
    trace = np.ones((10, 2))

    with pytest.raises(ValueError, match="one-dimensional"):
        rms_amplitude(trace)
