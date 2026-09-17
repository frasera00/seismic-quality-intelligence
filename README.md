# Seismic Quality Intelligence

> A gather-aware machine-learning workflow for synthetic seismic trace quality control, anomaly detection, threshold calibration, and interpretable prediction overlays.

## Overview

`seismic-quality-intelligence` is a recruiter-facing machine-learning portfolio project that applies unsupervised anomaly detection to seismic quality control (QC).

Rather than treating seismic traces as independent signals, the project generates and analyzes complete synthetic CMP-style gathers with coherent hyperbolic moveout. It then detects trace-level anomalies using amplitude, spectral, and local neighborhood-consistency features.

The project demonstrates:

- Synthetic seismic-gather generation with hyperbolic reflection events
- Injection of realistic trace-level QC anomalies
- Feature engineering for amplitudes, spectra, and neighboring-trace consistency
- Normal-only Isolation Forest anomaly detection
- Group-aware train, calibration, and validation splitting by gather
- Threshold calibration instead of relying only on the model default
- Anomaly-type recall analysis for dead traces, gain anomalies, and spikes
- Full-gather prediction overlays and uncertainty-aware score visualization
- Reproducible experiment artifacts and configuration-driven policy selection

## Why Gather-Aware QC?

A trace can appear statistically unusual for reasons that are not acquisition defects. In a seismic gather, neighboring traces share coherent events but differ with offset because of moveout, amplitude variation, and recording geometry.

This project therefore evaluates traces in their gather context:

```text
Synthetic gathers
    -> trace + neighborhood features
    -> group-aware anomaly detection
    -> calibrated policy selection
    -> interpretable seismic QC overlays
```

The group-aware protocol prevents leakage: traces from the same gather are never split between training, calibration, and final validation.

## Example Output

The workflow produces a full seismic-gather overlay containing:

- **Green:** correctly detected anomalies (true positives)
- **Orange:** normal traces flagged as anomalies (false positives)
- **Magenta:** anomalies not automatically flagged (false negatives)
- Isolation Forest decision score by trace offset
- A learned high-confidence alert threshold
- A configurable uncertainty-score band

 ![Calibrated seismic QC prediction overlay](docs/images/gather_prediction_overlay.png)

## Synthetic Data

The baseline experiment uses synthetic CMP-style gathers with hyperbolic arrivals:

\[
t(x) = \sqrt{t_0^2 + \frac{x^2}{v^2}}
\]

The current baseline configuration generates:

| Quantity | Value |
|---|---:|
| Total gathers | 120 |
| Traces per gather | 96 |
| Total trace samples | 11,520 |
| Approximate anomaly fraction | 10% |
| Outer training gathers | 90 |
| Held-out validation gathers | 30 |

### Injected anomaly types

| Type | Description | Main signals expected by the model |
|---|---|---|
| `dead_trace` | Strongly suppressed or absent trace | Low energy, reduced amplitude range, poor local consistency |
| `gain` | Amplitude-scaled but structurally coherent trace | Amplitude departure with partial similarity to normal seismic waveforms |
| `spike` | Localized impulsive disturbance | High kurtosis and elevated local residual |

## Features

Each trace is represented by seven features:

| Feature | Description |
|---|---|
| RMS amplitude | Overall signal energy |
| Peak-to-peak amplitude | Amplitude range |
| Kurtosis | Sensitivity to impulsive samples and spikes |
| Dominant frequency | Frequency of strongest spectral component |
| Spectral entropy | Spectral complexity / dispersion |
| Neighbor correlation | Similarity to an adjacent-trace reference |
| Neighbor residual RMS | Local mismatch relative to neighboring traces |

The model uses both trace-level characteristics and local spatial consistency, which makes the output more interpretable in seismic context.

## Model and Validation

### Model

- **Algorithm:** Isolation Forest
- **Training population:** normal traces only
- **Scaling:** `StandardScaler`, fitted using normal detector-training traces only
- **Baseline forest:** 300 estimators

Isolation Forest decision scores are interpreted as:

```text
More negative -> more anomalous
More positive -> more normal
```

### Split design

```text
120 total gathers
    |
    +-- 90 outer-training gathers
    |       |
    |       +-- 67 detector-training gathers
    |       +-- 23 calibration gathers
    |
    +-- 30 untouched validation gathers
```

`GroupShuffleSplit` uses `gather_id` as a group identifier. This prevents traces from a single coherent gather from appearing in both detector training and evaluation partitions.

## Baseline Results

### Default Isolation Forest threshold

The default decision boundary flags a trace when:

```text
score < 0.0
```

| Metric | Validation result |
|---|---:|
| Precision | 0.5455 |
| Recall | 1.0000 |
| F1 | 0.7059 |
| ROC AUC | 0.9895 |
| PR AUC | 0.8791 |
| True positives | 300 |
| False positives | 250 |
| True negatives | 2,330 |
| False negatives | 0 |

The default model detects all injected validation anomalies but creates a large number of false-positive alerts.

### Calibrated threshold

A stricter threshold was selected on independent calibration gathers to maximize precision while maintaining high overall recall.

| Metric | Default threshold | Strict calibrated threshold |
|---|---:|---:|
| Precision | 0.5455 | 0.7599 |
| Recall | 1.0000 | 0.9600 |
| F1 | 0.7059 | 0.8483 |
| True positives | 300 | 288 |
| False positives | 250 | 91 |
| False negatives | 0 | 12 |

This calibration reduced false positives by approximately **63.6%**, from 250 to 91, while retaining 96% aggregate recall.

## Anomaly-Type Insight

Aggregate results alone can hide important domain-specific behavior.

| Anomaly type | Validation count | Default recall | Strict-threshold recall |
|---|---:|---:|---:|
| `dead_trace` | 106 | 1.0000 | 1.0000 |
| `gain` | 99 | 1.0000 | 0.8990 |
| `spike` | 95 | 1.0000 | 0.9789 |

The stricter global threshold improves precision, but gain anomalies are more likely to fall near the decision boundary because they can preserve coherent seismic waveform structure.

This led to a **gain-safe calibration policy**: a candidate threshold can be selected by maximizing precision while enforcing both overall-recall and gain-recall constraints.

## Operating Policy

The project separates static experiment requirements from learned deployment values.

### Static configuration

`configs/baseline.yaml` stores experiment settings and calibration requirements:

```yaml
operating_policy:
  name: gain_safe_precision
  overall_recall_target: 0.95
  gain_recall_target: 0.98
  uncertainty_threshold: 0.0
```

### Learned policy artifact

After calibration, the learned policy is written to:

```text
reports/operating_policy.yaml
```

It contains the current learned alert threshold and the configured uncertainty threshold. Plotting and downstream inference scripts should read this file automatically rather than hard-coding numeric thresholds.

### Three-band score interpretation

```text
score < learned alert threshold
    High-confidence anomaly alert

learned alert threshold <= score < uncertainty threshold
    Uncertain score band

score >= uncertainty threshold
    Normal-confidence trace
```

The uncertainty band prevents borderline traces from being presented as confidently normal while retaining a high-confidence anomaly category.

## Repository Layout

```text
seismic-quality-intelligence/
├── configs/
│   └── baseline.yaml
├── docs/
│   ├── seismic_qc_baseline_report.md
│   └── project_handoff.md
├── reports/
│   ├── gather_qc_model.joblib
│   ├── gather_qc_metrics.yaml
│   ├── calibrated_thresholds.yaml
│   ├── operating_policy.yaml
│   └── figures/
│       └── gather_prediction_overlay.png
├── scripts/
│   ├── train.py
│   ├── calibrate_threshold.py
│   ├── plot_gather_examples.py
│   └── plot_gather_predictions.py
└── src/
    └── seismic_quality_intelligence/
        ├── synthetic_gather.py
        ├── gather_features.py
        ├── detectors.py
        └── evaluation.py
```

> `reports/` is normally generated locally and should be excluded from version control, except for deliberately curated small example images copied into `docs/images/`.

## Installation

### 1. Clone the repository

```powershell
git clone <your-repository-url>
cd seismic-quality-intelligence
```

### 2. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

Install the project dependencies using the repository dependency file if present:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If a `requirements.txt` file has not yet been created, install the baseline dependencies:

```powershell
python -m pip install numpy scipy matplotlib scikit-learn pyyaml joblib ruff
```

## Run the Baseline

From the repository root:

```powershell
python scripts\train.py --config configs\baseline.yaml
python scripts\calibrate_threshold.py --config configs\baseline.yaml --output-dir reports
python scripts\plot_gather_predictions.py
```

The plotting script selects a random gather from the held-out validation partition by default.

To inspect one specific validation gather:

```powershell
python scripts\plot_gather_predictions.py --gather-index 1
```

## Code Quality Checks

```powershell
python -m ruff format src\seismic_quality_intelligence\gather_features.py scripts\train.py scripts\calibrate_threshold.py scripts\plot_gather_predictions.py
python -m ruff check src\seismic_quality_intelligence\gather_features.py scripts\train.py scripts\calibrate_threshold.py scripts\plot_gather_predictions.py
```

## Current Limitations

This is a synthetic-data baseline and does not establish performance on field seismic data.

Known limitations and active investigation areas include:

- Synthetic anomalies may be more regular than real acquisition failures.
- Current anomaly classes are limited to dead traces, gain changes, and spikes.
- Neighbor-consistency features are not yet aligned for local moveout.
- Visual inspection suggests that normal far-offset traces may receive lower anomaly scores.
- The offset-score pattern is a hypothesis, not a confirmed causal result, until it is tested using deterministic offset-binned diagnostics on normal traces.
- Threshold selection depends on the stated cost of false positives versus missed anomaly types.

## Next Steps

1. Quantify normal-trace score behavior against absolute offset.
2. Add normalized neighbor residual features.
3. Evaluate normalized absolute offset as a model feature.
4. Implement lag-aligned neighbor correlation and residual measures.
5. Compare default, F1-optimal, precision-at-recall, and gain-safe operating policies.
6. Evaluate the methodology on more realistic simulations or field-data QC examples.
7. Add optional local LLM-based experiment reporting after validating the deterministic reporting workflow.

## Detailed Documentation

- [Baseline technical report](docs/seismic_qc_baseline_report.md)
- [Project handoff and development notes](docs/project_handoff.md)

## Portfolio Statement

> Built an end-to-end seismic trace-quality intelligence workflow using synthetic CMP-style gathers, spectral and local-consistency features, group-aware unsupervised anomaly detection, subtype-aware threshold calibration, and interpretable prediction overlays. The project emphasizes reproducible evaluation, calibration discipline, domain-informed error analysis, and uncertainty-aware model outputs.

## License

Add an appropriate license before publishing publicly. For a portfolio project, the MIT License is a common permissive choice.
