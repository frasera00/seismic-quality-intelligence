# Model Card — Synthetic Seismic Gather QC Detector

## Model Summary

This project uses an Isolation Forest to identify anomalous seismic traces
within synthetic CMP-style gathers containing hyperbolic moveout events.

The detector is trained on normal traces only. Synthetic injected anomalies are
used exclusively for controlled calibration and held-out evaluation.

## Intended Use

- Educational and portfolio demonstration of seismic QC anomaly detection.
- Trace-level screening of synthetic prestack-style gathers.
- Comparison of anomaly-score thresholds and subtype-specific recall.
- Visualization of predictions in seismic gather context.

## Not Intended For

- Direct deployment on field seismic data.
- QC decisions on stacked, migrated, zero-offset, or spatial-profile data
  without retraining and revalidation.
- Production acquisition decisions.
- Performance claims on real seismic surveys.

## Data

The baseline dataset contains 120 synthetic gathers with 96 traces per gather.
The injected anomaly classes are:

- `dead_trace`
- `gain`
- `spike`

The outer split uses groups defined by gather ID:

```text
90 outer-training gathers
30 held-out validation gathers
```

The outer-training partition is further split into detector-training and
threshold-calibration gathers.

## Features

- RMS amplitude
- Peak-to-peak amplitude
- Kurtosis
- Dominant frequency
- Spectral entropy
- Neighbor correlation
- Neighbor residual RMS

## Training Procedure

1. Generate deterministic synthetic gathers using the configured seed.
2. Split gathers with `GroupShuffleSplit`.
3. Fit `StandardScaler` on normal detector-training traces only.
4. Train Isolation Forest on scaled normal detector-training traces.
5. Select a threshold only on separate calibration gathers.
6. Report final results on untouched outer-validation gathers.

## Baseline Results

### Default threshold

```text
Threshold: 0.0
Precision: 0.5455
Recall:    1.0000
F1:        0.7059
TP / FP / TN / FN: 300 / 250 / 2330 / 0
ROC AUC: 0.9895
PR AUC:  0.8791
```

### Strict calibrated policy

```text
Precision: 0.7599
Recall:    0.9600
F1:        0.8483
TP / FP / TN / FN: 288 / 91 / 2489 / 12
```

## Important Subtype Finding

The stricter calibrated threshold improves aggregate precision and F1 but is
less sensitive to `gain` anomalies than to dead traces and spikes.

```text
Dead-trace recall: 1.0000
Gain recall:       0.8990
Spike recall:      0.9789
```

Consequently, a single strict global threshold should not be treated as
universally optimal. The project supports gain-aware threshold constraints and
an uncertainty score band.

## Limitations

- All quantitative results use synthetic data.
- Anomaly types are simplified relative to field acquisition failures.
- Neighbor features are not yet moveout-aligned.
- Normal traces may show offset-dependent score behavior.
- The model has not been validated on raw multichannel field gathers.
- A real zero-offset profile is a different data representation and requires
  a separate training and evaluation workflow.

## Reproducibility

Run:

```
python scripts\train.py --config configs\baseline.yaml
python scripts\calibrate_threshold.py --config configs\baseline.yaml
python scripts\plot_gather_predictions.py
```

See `README.md` and `docs/seismic_qc_baseline_report.md` for full details.