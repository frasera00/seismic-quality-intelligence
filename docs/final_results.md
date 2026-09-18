# Final Results

## Evaluation protocol

- 120 synthetic gathers
- Group-aware outer split: 90 training / 30 held-out validation gathers
- Threshold calibration on separate training gathers
- Normal-only Isolation Forest training

## Default high-sensitivity baseline

| Metric | Value |
|---|---:|
| Precision | 0.5455 |
| Recall | 1.0000 |
| F1 | 0.7059 |
| TP / FP / TN / FN | 300 / 250 / 2330 / 0 |

## Strict calibrated policy

| Metric | Value |
|---|---:|
| Precision | 0.7599 |
| Recall | 0.9600 |
| F1 | 0.8483 |
| TP / FP / TN / FN | 288 / 91 / 2489 / 12 |

## Main conclusion

Calibration reduced false positives by 63.6% while retaining 96.0% aggregate
recall. However, the strict global threshold reduced gain-anomaly recall, so
the system should use an anomaly-type-aware operating policy and uncertainty
band rather than presenting a single threshold as universally optimal.