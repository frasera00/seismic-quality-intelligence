import numpy as np


def rms_amplitude(trace: np.ndarray) -> float:
    """Compute the root-mean-square amplitude of a one-dimensional signal."""
    trace = np.asarray(trace, dtype=float)

    if trace.ndim != 1:
        raise ValueError("trace must be a one-dimensional array")

    if trace.size == 0:
        raise ValueError("trace must not be empty")

    return float(np.sqrt(np.mean(trace**2)))
