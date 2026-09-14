# Seismic Quality Intelligence — Project Handoff

**Last updated:** 2026-09-14  
**Project purpose:** Recruiter-facing portfolio project demonstrating machine learning, anomaly detection, geophysical QC, threshold calibration, visualization, reproducibility, and planned local-LLM reporting.

## 1. Project Summary

This project builds a synthetic seismic-gather quality-control system that detects anomalous traces in CMP-style gathers containing hyperbolic moveout events.

The intended workflow is:

```text
Synthetic seismic gathers
    -> trace and neighborhood feature extraction
    -> group-aware train / calibration / validation split
    -> normal-only Isolation Forest training
    -> threshold calibration with anomaly-type constraints
    -> prediction overlays on full seismic gathers
    -> automated experiment documentation
    -> optional local LLM-generated analysis
```

The project is deliberately designed around **whole seismic gathers**, rather than independent traces, so that QC predictions can be interpreted in acquisition and moveout context.

## 2. Current Data and Split

### Synthetic data

Current baseline configuration generates:

```text
120 synthetic gathers
96 traces per gather
11,520 total trace samples
Approximately 10% injected anomalous traces
```

Injected anomaly types:

- `dead_trace`
- `gain`
- `spike`

### Evaluation protocol

The outer split is group-aware by `gather_id`:

```text
90 outer-training gathers
30 untouched validation gathers
```

The 90 outer-training gathers are split again:

```text
67 detector-training gathers
23 calibration gathers
```

This prevents trace-level leakage: traces from the same gather must never be split across detector training, calibration, or outer validation.

## 3. Features and Model

### Trace features

The current gather-level feature extractor uses seven features per trace:

1. RMS amplitude
2. Peak-to-peak amplitude
3. Kurtosis
4. Dominant frequency
5. Spectral entropy
6. Neighbor correlation
7. Neighbor residual RMS

The neighborhood features compare each trace against a reference derived from adjacent traces.

### Model

```text
Model: Isolation Forest
Training: normal traces only
Scaling: StandardScaler fitted on normal detector-training traces
```

The model produces Isolation Forest decision scores:

```text
More negative score -> more anomalous
More positive score -> more normal
```

## 4. Baseline Results

### Default Isolation Forest threshold

The default score boundary is:

```text
score < 0.0 -> anomaly
```

Outer-validation results previously obtained:

| Metric | Result |
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

Interpretation:

- The detector ranks anomalies strongly.
- The default threshold catches all injected validation anomalies.
- The default threshold creates many false-positive trace flags.

## 5. Threshold Calibration Findings

A calibration script selects thresholds from separate calibration gathers rather than using validation gathers for threshold selection.

A stricter threshold selected for aggregate precision at high recall produced approximately:

| Metric | Default threshold | Strict calibrated threshold |
|---|---:|---:|
| Precision | 0.5455 | 0.7599 |
| Recall | 1.0000 | 0.9600 |
| F1 | 0.7059 | 0.8483 |
| TP | 300 | 288 |
| FP | 250 | 91 |
| FN | 0 | 12 |

The strict threshold reduced false positives substantially but introduced a domain-relevant problem: it missed a disproportionate number of `gain` anomalies.

### Anomaly-type recall finding

For one strict calibrated policy, outer-validation subtype results were:

| Type | Count | Default recall | Strict-policy recall |
|---|---:|---:|---:|
| `dead_trace` | 106 | 1.0000 | 1.0000 |
| `gain` | 99 | 1.0000 | 0.8990 |
| `spike` | 95 | 1.0000 | 0.9789 |

Key conclusion:

> Aggregate recall can conceal subtype-specific failure. A single global threshold is too aggressive if gain anomalies are operationally important.

## 6. Operating Policy

The calibration workflow is being converted from a manually copied threshold into an automatic operating-policy artifact.

### Static policy requirements

Configure policy requirements in:

```text
configs/baseline.yaml
```

Expected configuration block:

```yaml
operating_policy:
  name: gain_safe_precision
  overall_recall_target: 0.95
  gain_recall_target: 0.98
  uncertainty_threshold: 0.0
```

The configuration stores requirements, not the learned alert threshold.

### Dynamic policy artifact

Each calibration run should write:

```text
reports/operating_policy.yaml
```

The policy artifact should contain:

```yaml
policy_name: gain_safe_precision
threshold: <learned calibrated alert threshold>
uncertainty_threshold: 0.0
decision_rule: flag when decision_score < threshold
uncertainty_rule: uncertain when threshold <= decision_score < uncertainty_threshold
```

Downstream plotting and inference code should read `reports/operating_policy.yaml` automatically. Do not hard-code a learned score threshold into a Python script or modify `baseline.yaml` with learned values.

### Current triage interpretation

```text
score < learned alert threshold
    High-confidence anomaly alert

learned alert threshold <= score < uncertainty threshold
    Uncertain score band

score >= uncertainty threshold
    Normal-confidence trace
```

The term **uncertain score band** is preferred over “manual review” because this is a recruiter-facing technical portfolio project. It communicates calibrated uncertainty rather than an unfinished operational workflow.

## 7. Current Scripts and Artifacts

### Primary scripts

```text
scripts/train.py
scripts/calibrate_threshold.py
scripts/plot_gather_predictions.py
scripts/plot_gather_examples.py
```

### Core source modules

```text
src/seismic_quality_intelligence/gather_features.py
src/seismic_quality_intelligence/synthetic_gather.py
src/seismic_quality_intelligence/detectors.py
src/seismic_quality_intelligence/evaluation.py
```

### Expected report artifacts

```text
reports/gather_qc_model.joblib
reports/gather_qc_metrics.yaml
reports/validation_predictions.npz
reports/calibrated_thresholds.yaml
reports/operating_policy.yaml
reports/figures/gather_prediction_overlay.png
```

## 8. Plotting Status

`plot_gather_predictions.py` should:

- Regenerate the synthetic dataset deterministically from `configs/baseline.yaml`.
- Reproduce the group-aware outer validation split.
- Load the trained model from `reports/gather_qc_model.joblib`.
- Load the learned policy from `reports/operating_policy.yaml`.
- Select a **random validation gather** by default.
- Support `--gather-index` to force a specific validation gather.
- Overlay TP, FP, and FN predictions on the seismic gather.
- Show Isolation Forest decision scores below the gather.
- Show the learned alert threshold and configured uncertainty threshold.
- Shade the uncertain score band between those thresholds.
- Mark samples in the uncertainty band distinctly.

### Current overlay conventions

```text
Green vertical line: true positive
Orange vertical line: false positive
Magenta vertical line: false negative
```

Known anomalous traces can be labeled near the bottom of the image. Labels should not overlap panel titles.

### Spike visibility improvement

Spikes can be visually subtle in a full-gather image because they occupy few samples. The plot can add an `x` marker at the time of the largest absolute amplitude on known spike traces.

Potential helper:

```python
def spike_time(trace: np.ndarray, time: np.ndarray) -> float:
    """Return the time of the largest absolute-amplitude sample."""
    sample_index = int(np.argmax(np.abs(trace)))
    return float(time[sample_index])
```

## 9. Important Open Finding: Offset-Dependent Score Drift

Visual inspection suggested that normal-trace Isolation Forest decision scores become lower toward large absolute offsets.

This is currently a **hypothesis requiring deterministic confirmation**, not a proven conclusion.

Plausible contributing mechanisms include:

- Hyperbolic moveout changes trace waveform position with offset.
- Far-offset events may approach the recording-window boundary.
- Unaligned neighbor correlation can decline as trace-to-trace moveout changes.
- Raw neighbor residual RMS can increase because adjacent traces differ more at large offsets.
- Edge traces use a one-sided reference rather than a two-sided neighbor reference.
- In real data, additional offset-dependent behavior can arise from amplitude variation, attenuation, and NMO stretch.

### Recommended deterministic diagnostic

Create an offset-analysis script that evaluates **normal traces only** and writes a table/figure with:

```text
absolute offset bin
trace count
mean and median IF score
score standard deviation
high-confidence alert rate
uncertain-band rate
mean neighbor correlation
mean neighbor residual RMS
```

Also compute:

```text
Pearson correlation: score vs absolute offset
Spearman correlation: score vs absolute offset
Near-offset vs far-offset median-score difference
Near-offset vs far-offset alert-rate difference
```

This should be evidence used in future reporting; do not rely on visual inspection alone.

## 10. Candidate Model Improvements

Prioritize experiments in this order:

1. **Offset diagnostics:** quantify score drift before changing the model.
2. **Normalized neighbor residual:** replace or complement raw neighbor residual RMS with:

   ```text
   RMS(trace - neighbor_reference) / (RMS(trace) + epsilon)
   ```

3. **Normalized absolute offset feature:** supply gather position to the model so it can learn expected edge behavior.
4. **Moveout-aware neighborhood features:** estimate local lag between a trace and neighbors, align neighbors, then calculate correlation/residual coherence.
5. **Gain-safe threshold calibration:** maximize precision subject to both overall-recall and gain-recall constraints.
6. **Compare policies:** default sensitivity screen, high-confidence threshold, F1-optimal threshold, and gain-safe threshold.

Each experiment should have a measurable acceptance criterion. Example:

```text
Reduce far-offset normal alert rate without lowering gain recall below target.
```

## 11. LLM Reporting Plan

The objective is to build a reporting layer in which an LLM independently reviews saved metrics, calibration results, policy artifacts, and figure(s), then writes an evidence-grounded Markdown report.

Do **not** tell the LLM in advance to inspect offsets, gain anomalies, or any specific trend. It should independently identify relevant patterns.

### Input artifacts

```text
configs/baseline.yaml
reports/gather_qc_metrics.yaml
reports/calibrated_thresholds.yaml
reports/operating_policy.yaml
reports/figures/gather_prediction_overlay.png
```

### Desired report structure

```markdown
# Experiment Analysis

## Summary

## Observations

## Interpretations

## Limitations

## Recommended Experiments
```

### Guardrails for the LLM

The prompt should require that the model:

- Treat saved numeric metrics as authoritative.
- Not invent metrics, model features, anomaly types, or processing steps.
- Separate direct observations from hypotheses/interpretations.
- State evidence and uncertainty for each interpretation.
- Not infer causal mechanisms solely from a visual trend.
- Not claim real field-data generalization from synthetic data.
- Propose only concrete follow-up experiments with measurable success criteria.

### Local LLM direction

The preferred implementation is Ollama with a vision-capable local model, so that reports can be generated without paid cloud API credits and without transmitting project artifacts externally.

Possible script name:

```text
scripts/generate_local_llm_report.py
```

Expected local dependencies:

```text
Ollama Desktop/runtime installed on the machine
A pulled vision model, for example llama3.2-vision
Python package: ollama
```

### Governance constraint

Do **not** install the Ollama runtime on the managed work computer without explicit company IT/security approval.

Although local inference may keep prompts and figure content local, Ollama installation still involves:

- Installing new runtime software.
- Downloading large model weights.
- Running a localhost service.
- Using corporate storage, memory, CPU/GPU, and network resources.

Safer options:

- Use a personal computer for Ollama.
- Request IT/security approval.
- Keep the LLM module optional and include a sanitized example report in the repository.

Note: Installing `ollama` with `pip` only installs the Python client. It does not install the Ollama runtime, `ollama.exe`, a model, or the local server.

## 12. Reproduction Commands

From the repository root:

```powershell
python -m ruff format src\seismic_quality_intelligence\gather_features.py scripts\train.py scripts\calibrate_threshold.py scripts\plot_gather_predictions.py
python -m ruff check src\seismic_quality_intelligence\gather_features.py scripts\train.py scripts\calibrate_threshold.py scripts\plot_gather_predictions.py

python scripts\train.py --config configs\baseline.yaml
python scripts\calibrate_threshold.py --config configs\baseline.yaml --output-dir reports
python scripts\plot_gather_predictions.py
```

To plot a specific validation gather:

```powershell
python scripts\plot_gather_predictions.py --gather-index 1
```

To generate an optional local LLM report after Ollama is approved and installed:

```powershell
python -m pip install ollama
python scripts\generate_local_llm_report.py
```

## 13. Immediate Next Steps

1. Confirm the latest `calibrate_threshold.py` writes both `calibrated_thresholds.yaml` and `operating_policy.yaml`.
2. Confirm `plot_gather_predictions.py` reads `reports/operating_policy.yaml` automatically.
3. Confirm the plot defaults to a random validation gather and accepts `--gather-index`.
4. Create a deterministic normal-trace offset-drift diagnostic before changing features.
5. Build a deterministic Markdown report generator.
6. Keep the Ollama LLM-reporting layer optional until using a personal machine or receiving approval for the work computer.
7. Add a polished README, architecture diagram, and selected experiment reports for the public GitHub portfolio.

## 14. Recruiter-Facing Project Description

> Built an end-to-end seismic trace-quality intelligence pipeline using synthetic CMP-style gathers, spectral and local-coherence features, group-aware unsupervised anomaly detection, subtype-constrained threshold calibration, and interpretable seismic prediction overlays. The project records calibrated operating policies and is being extended with automated, evidence-grounded experiment reporting and optional local multimodal LLM analysis.

## 15. Important Principles to Preserve

- Keep `configs/baseline.yaml` as static experiment input; do not overwrite it with learned threshold values.
- Write learned thresholds and policy metadata to `reports/operating_policy.yaml`.
- Never choose thresholds using the untouched outer-validation set.
- Report anomaly-type recall, not aggregate recall alone.
- Treat figures as interpretability artifacts, but verify visual trends with deterministic numeric diagnostics.
- Keep local-LLM reporting optional and do not commit keys, credentials, private data, or large model files to Git.
