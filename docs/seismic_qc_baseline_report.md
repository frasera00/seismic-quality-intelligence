# Seismic Quality Intelligence
## Gather-Level Anomaly Detection Baseline Report

**Project status:** Baseline complete  
**Data:** Synthetic seismic CMP-style gathers  
**Model:** Normal-only Isolation Forest  
**Primary objective:** Detect trace-level quality-control anomalies while preserving gather-level seismic context

---

## Executive Summary

This project implements a seismic trace-quality-control workflow using synthetic common-midpoint-style gathers with hyperbolic reflection events. The workflow generates labeled gathers, extracts trace-level and neighborhood-consistency features, trains an unsupervised Isolation Forest using only normal traces, calibrates its score threshold using separate calibration gathers, and visualizes predictions directly on full seismic gathers.

The initial default Isolation Forest decision boundary achieved perfect recall on the held-out validation set: all 300 injected anomalies were detected. However, this high-sensitivity setting also produced 250 false-positive trace alerts. Calibration on an independent set of gathers improved validation precision from 54.5% to approximately 76.0% and F1 from 0.706 to 0.848, reducing false positives to 91. The stricter threshold introduced a meaningful domain-specific trade-off: it reduced recall for gain anomalies more than for dead traces or spikes.

The project therefore evolved from a single binary anomaly threshold into a policy-based design: a learned high-confidence alert threshold is paired with a configurable uncertainty threshold. This makes score ambiguity explicit and prevents borderline traces from being presented as confidently normal.

---

## 1. Problem Definition

Seismic acquisition quality control aims to identify traces that may be unreliable before they influence subsequent processing, imaging, inversion, or interpretation. Common trace-level defects include dead channels, abnormal gain, isolated spikes, excessive noise, timing issues, and poor local coherence.

The objective of this baseline is to answer the following question:

> Can an unsupervised anomaly-detection model identify synthetic trace-level QC failures in seismic gathers while preserving the spatial and moveout context needed for geophysical interpretation?

The workflow uses an Isolation Forest because it can be trained from a population of normal traces without requiring a large labeled database of real acquisition failures. Synthetic labels are still used for controlled calibration and evaluation.

---

## 2. Synthetic Gather Design

### 2.1 Gather representation

The synthetic dataset uses seismic gathers with multiple coherent hyperbolic reflection events. The events follow normal-moveout-like kinematics:

\[
t(x) = \sqrt{t_0^2 + \frac{x^2}{v^2}}
\]

where:

- \(t(x)\) is arrival time at offset \(x\)
- \(t_0\) is zero-offset time
- \(v\) is an effective velocity

This design creates a more realistic QC setting than isolated, independent traces because the signal changes with offset and neighboring traces are related through coherent moveout.

### 2.2 Baseline dataset

The baseline experiment generated:

| Quantity | Value |
|---|---:|
| Total gathers | 120 |
| Traces per gather | 96 |
| Total trace samples | 11,520 |
| Approximate anomaly fraction | 10% |
| Outer training gathers | 90 |
| Outer validation gathers | 30 |

Each gather contains coherent events, noise, acquisition-style offsets, and injected trace anomalies.

### 2.3 Injected anomaly types

The synthetic generator produces three anomaly classes:

| Anomaly type | Description | Expected feature effects |
|---|---|---|
| `dead_trace` | A trace with substantially suppressed or absent signal | Low RMS and peak-to-peak amplitude; weak neighborhood agreement |
| `gain` | An amplitude-scaled trace that remains structurally coherent | Elevated amplitude statistics; potentially similar frequency and neighbor correlation |
| `spike` | A localized impulsive disturbance | Increased kurtosis and local neighbor residual |

The gain anomaly is intentionally more difficult than a dead trace or an isolated spike because it can preserve much of the original seismic waveform and moveout structure.

---

## 3. Feature Engineering

Each trace is transformed into a seven-dimensional feature vector.

| Feature | Purpose |
|---|---|
| RMS amplitude | Measures overall trace energy |
| Peak-to-peak amplitude | Captures amplitude range and gain excursions |
| Kurtosis | Detects impulsive amplitude behavior such as spikes |
| Dominant frequency | Summarizes spectral concentration |
| Spectral entropy | Measures spectral complexity or dispersion |
| Neighbor correlation | Measures waveform similarity to adjacent traces |
| Neighbor residual RMS | Measures local mismatch relative to neighboring traces |

The final two features add gather context. Rather than deciding from one trace alone, the model can use local consistency with neighboring traces.

For interior traces, the neighbor reference is constructed from immediate adjacent traces. At gather boundaries, the closest available neighbor is used.

---

## 4. Model Training

### 4.1 Preprocessing

A `StandardScaler` is fitted using normal detector-training traces only. The same fitted scaler then transforms calibration and validation data.

This ensures that model scaling is derived from the intended normal reference population and avoids using validation statistics during fitting.

### 4.2 Isolation Forest

The detector is an Isolation Forest configured with:

- 300 estimators
- Fixed random seed for reproducibility
- Contamination parameter of 0.10
- Training on normal traces only

The Isolation Forest produces a decision score for every trace:

```text
More negative score  -> more anomalous
More positive score  -> more normal
```

The default classification boundary is:

```text
decision score < 0.0 -> anomalous
```

### 4.3 Group-aware splitting

A critical requirement is that traces from the same gather are never shared across detector training, calibration, and validation partitions.

The project uses `GroupShuffleSplit` with `gather_id` as the group identifier.

The split hierarchy is:

```text
120 total gathers
    |
    +-- 90 outer-training gathers
    |       |
    |       +-- 67 detector-training gathers
    |       +-- 23 threshold-calibration gathers
    |
    +-- 30 untouched outer-validation gathers
```

This approach prevents optimistic results caused by traces from the same coherent gather appearing on both sides of a split.

---

## 5. Default Threshold Results

The initial baseline used the Isolation Forest default decision boundary of zero.

### 5.1 Held-out validation results

| Metric | Value |
|---|---:|
| Precision | 0.5455 |
| Recall | 1.0000 |
| F1 score | 0.7059 |
| ROC AUC | 0.9895 |
| PR AUC | 0.8791 |
| True positives | 300 |
| False positives | 250 |
| True negatives | 2,330 |
| False negatives | 0 |

### 5.2 Interpretation

The model successfully ranked anomalous traces relative to normal traces, as indicated by ROC AUC of 0.9895 and PR AUC of 0.8791. It detected all 300 injected anomalies in the outer-validation set.

However, the default threshold flagged 550 traces as anomalous:

```text
300 true anomalies
250 false positives
```

This gave perfect recall but modest precision. The result is suitable as a high-sensitivity screening baseline, but it would create unnecessary alerts in a practical workflow.

---

## 6. Threshold Calibration

### 6.1 Why calibration was needed

The Isolation Forest default threshold is a generic model boundary, not necessarily the best operational decision threshold for seismic QC. The score distribution showed substantial ranking power, but some normal traces fell below zero.

Threshold calibration was therefore performed on 23 dedicated calibration gathers, which were separate from both detector training and untouched outer validation.

### 6.2 Calibration policies

The calibration script evaluates multiple policies:

| Policy | Selection criterion |
|---|---|
| Default | Fixed Isolation Forest boundary: score < 0.0 |
| F1-optimal | Calibration threshold that maximizes F1 |
| Precision at recall | Calibration threshold that maximizes precision while satisfying an overall recall target |
| Gain-safe precision | Calibration threshold that maximizes precision while satisfying overall and gain-specific recall targets |

The precision-at-recall policy initially used an overall recall target of 0.95.

### 6.3 Strict calibrated-threshold results

One strict calibration policy selected a threshold near:

```text
-0.11985
```

Its outer-validation performance was:

| Metric | Default threshold | Strict calibrated threshold |
|---|---:|---:|
| Precision | 0.5455 | 0.7599 |
| Recall | 1.0000 | 0.9600 |
| F1 score | 0.7059 | 0.8483 |
| True positives | 300 | 288 |
| False positives | 250 | 91 |
| True negatives | 2,330 | 2,489 |
| False negatives | 0 | 12 |

### 6.4 Improvement from calibration

Relative to the default boundary, the calibrated threshold:

- Reduced false positives by 159 traces, from 250 to 91.
- Reduced false positives by approximately 63.6%.
- Increased precision by approximately 21.4 percentage points.
- Increased F1 from 0.706 to 0.848.
- Retained 96.0% aggregate recall.

This demonstrates that score calibration can substantially improve the practical usefulness of an anomaly detector without changing the underlying Isolation Forest model.

---

## 7. Anomaly-Type Analysis

Aggregate metrics alone were insufficient because different anomaly types behaved differently.

### 7.1 Validation recall by anomaly type

For one strict calibrated policy, validation results were:

| Anomaly type | Validation count | Default recall | Strict-policy recall |
|---|---:|---:|---:|
| `dead_trace` | 106 | 1.0000 | 1.0000 |
| `gain` | 99 | 1.0000 | 0.8990 |
| `spike` | 95 | 1.0000 | 0.9789 |

### 7.2 Key result

The strict global threshold preserved all dead-trace detections and nearly all spike detections, but missed approximately 10 of 99 gain anomalies.

This finding is important because gain anomalies can remain coherent with the seismic signal. They may resemble legitimate amplitude variation more than dead channels or impulsive spikes. As a result, a threshold chosen only to optimize aggregate precision or F1 can penalize a domain-important subtype.

### 7.3 Implication

The project should not treat a single strict global threshold as universally correct. Threshold selection must reflect the relative operational cost of missing each anomaly type.

The `gain_safe_precision` policy was introduced to support calibration that maximizes precision while requiring both:

```text
overall recall >= configured target
gain-anomaly recall >= configured target
```

---

## 8. Operating Policy and Uncertainty Band

### 8.1 Static configuration

The static experiment configuration is stored in:

```text
configs/baseline.yaml
```

The configuration defines policy requirements rather than a fixed learned threshold. Example:

```yaml
operating_policy:
  name: gain_safe_precision
  overall_recall_target: 0.95
  gain_recall_target: 0.98
  uncertainty_threshold: 0.0
```

### 8.2 Learned artifact

Each calibration run writes a policy artifact:

```text
reports/operating_policy.yaml
```

This file is the source of truth for downstream prediction and plotting scripts. It records:

- Selected policy name
- Learned high-confidence alert threshold
- Configured uncertainty threshold
- Decision rule
- Calibration constraints
- Calibration and validation metrics
- Anomaly-type recall information

This separation is intentional:

```text
baseline.yaml
    Static input configuration and policy requirements

operating_policy.yaml
    Learned dynamic threshold and evaluated operating policy
```

### 8.3 Three-band decision design

The scoring policy is represented as:

```text
score < learned alert threshold
    High-confidence anomaly alert

learned alert threshold <= score < uncertainty threshold
    Uncertain score band

score >= uncertainty threshold
    Normal-confidence trace
```

The uncertain score band is preferable to presenting every trace as a confident binary decision. It makes the threshold trade-off visible and is particularly relevant for gain anomalies close to the strict alert boundary.

---

## 9. Prediction Visualization

The gather-prediction visualization overlays model outcomes onto a full synthetic seismic gather.

### 9.1 Plot contents

The upper panel shows:

- Seismic gather amplitudes using a diverging colormap
- Hyperbolic reflection events
- Known anomaly labels near the bottom of the panel
- Vertical classification markers

The lower panel shows:

- Isolation Forest decision score by offset
- Learned alert threshold
- Configured uncertainty threshold
- Shaded uncertain score band
- Score points colored by classification outcome

### 9.2 Outcome colors

| Color | Meaning |
|---|---|
| Green | True positive |
| Orange | False positive |
| Magenta | False negative |
| Blue | True negative score point |

Spike anomalies can also be annotated with an `x` marker at the largest absolute-amplitude sample, making localized spikes easier to find in the full gather display.

### 9.3 Random validation gathers

The plotting script selects a random gather from the held-out validation set by default. A specific gather can still be requested with:

```powershell
python scripts\plot_gather_predictions.py --gather-index 1
```

This supports both broad visual inspection and repeatable debugging.

---

## 10. Observed Limitation: Offset-Dependent Score Behavior

Visual inspection of score plots suggested that normal traces at large absolute offsets can receive lower Isolation Forest scores than central traces.

This observation is not yet a confirmed causal result. It should be tested with a dedicated deterministic diagnostic using normal traces only.

### 10.1 Plausible causes

Potential contributors include:

- Hyperbolic moveout changes the time position of coherent energy with offset.
- Neighboring traces differ more at far offsets because local moveout slope changes.
- Neighbor correlation and raw neighbor residual RMS are calculated without moveout alignment.
- Some events may approach the recording-window boundary at far offsets.
- Edge traces use one-sided rather than two-sided neighbor references.
- Global trace statistics can vary with gather geometry and offset.

### 10.2 Why it matters

If the model systematically assigns lower scores to normal far-offset traces, it may increase false-positive alerts at the gather edges. This is a potential acquisition-geometry covariate shift rather than a true defect pattern.

### 10.3 Required next diagnostic

A follow-up script should calculate, for normal validation traces only:

- Pearson and Spearman association between decision score and absolute offset
- Score mean, median, and spread by absolute-offset bin
- Alert rate by offset bin
- Uncertain-band rate by offset bin
- Neighbor correlation by offset bin
- Neighbor residual RMS by offset bin
- Near-offset versus far-offset score and alert-rate differences

This diagnostic must be computed numerically before any visual trend is described as a model bias or a confirmed physical mechanism.

---

## 11. Recommended Next Experiments

### 11.1 Quantify offset effects

**Change:** Implement deterministic score-versus-absolute-offset diagnostics using normal traces only.

**Success criterion:** Determine whether far-offset normal traces have materially lower median scores or higher alert rates than near-offset normal traces.

### 11.2 Normalize local residuals

**Change:** Add a scale-invariant residual feature:

\[
\frac{\operatorname{RMS}(s_i - s_{\mathrm{reference}})}
{\operatorname{RMS}(s_i) + \epsilon}
\]

**Success criterion:** Reduce sensitivity to ordinary amplitude variation while maintaining gain-anomaly recall.

### 11.3 Add normalized absolute offset

**Change:** Add normalized absolute offset as a feature.

**Success criterion:** Reduce edge-related normal-trace false positives without reducing anomaly detection performance.

### 11.4 Add lag-aligned neighbor coherence

**Change:** Estimate local time lag between a trace and its neighbors, align the reference trace, and then calculate correlation and residual features.

**Success criterion:** Improve far-offset normal-trace scores and reduce offset-dependent alert rates while preserving dead-trace, spike, and gain recall.

### 11.5 Compare calibrated operating policies

**Change:** Evaluate default, F1-optimal, precision-at-recall, and gain-safe precision policies side by side.

**Success criterion:** Select the policy that satisfies explicit subtype recall requirements at the lowest acceptable false-positive burden.

---

## 12. Reproducibility Workflow

The baseline workflow is run from the repository root.

```powershell
python -m ruff format src\seismic_quality_intelligence\gather_features.py scripts\train.py scripts\calibrate_threshold.py scripts\plot_gather_predictions.py
python -m ruff check src\seismic_quality_intelligence\gather_features.py scripts\train.py scripts\calibrate_threshold.py scripts\plot_gather_predictions.py

python scripts\train.py --config configs\baseline.yaml
python scripts\calibrate_threshold.py --config configs\baseline.yaml --output-dir reports
python scripts\plot_gather_predictions.py
```

Expected generated artifacts include:

```text
reports/gather_qc_model.joblib
reports/gather_qc_metrics.yaml
reports/validation_predictions.npz
reports/calibrated_thresholds.yaml
reports/operating_policy.yaml
reports/figures/gather_prediction_overlay.png
```

The learned threshold should always be read from `reports/operating_policy.yaml`; it should not be copied manually into plotting or inference code.

---

## 13. Scope and Limitations

This is a synthetic-data baseline. It demonstrates a controlled machine-learning and QC methodology but does not establish field-data performance.

Important limitations include:

- Synthetic anomalies may be easier or more regular than real acquisition failures.
- The current anomalies cover only dead traces, gain changes, and spikes.
- Noise, wavelet behavior, geometry, and subsurface structure remain simplified.
- Neighbor features are not yet moveout-aligned.
- The final chosen operating threshold depends on stated anomaly-type recall requirements.
- Results must be revalidated on representative field or realistic simulated data before operational use.

---

## 14. Conclusion

The project establishes an end-to-end, gather-aware seismic QC anomaly-detection baseline. It combines synthetic CMP-style gathers, trace-level spectral and local-consistency features, normal-only Isolation Forest training, group-aware validation, threshold calibration, subtype analysis, and interpretable seismic overlays.

The baseline model achieved excellent anomaly ranking and perfect recall under the default threshold. Calibration substantially reduced false positives and improved F1, but subtype analysis revealed that gain anomalies are more vulnerable to stricter global thresholds. This led to a more defensible operating-policy architecture that includes gain-aware calibration constraints and an uncertainty band.

The next phase should quantify the observed score dependence on absolute offset, improve moveout awareness in neighborhood features, and compare policy choices using explicit QC objectives. This progression demonstrates not only model training, but also error analysis, calibration discipline, domain-aware evaluation, reproducibility, and responsible uncertainty communication.
