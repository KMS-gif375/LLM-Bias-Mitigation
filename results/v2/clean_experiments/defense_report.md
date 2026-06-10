# Clean Defense Suite Report

This report is generated without LLM inference from saved signals/predictions.

## Same-Test-ID Aggregate Metrics

| System | Subset | Seeds | n | acc_amb | acc_dis | abs_bias_amb | FAR |
|---|---|---:|---:|---:|---:|---:|---:|
| composite | shared_test | 5 | 1328 | 0.6843±0.0138 | 0.2855±0.0109 | 0.0806±0.0685 | 0.2449±0.0164 |
| decap | shared_test | 5 | 1328 | 0.8057±0.0055 | 0.7238±0.0075 | 0.4296±0.0486 | 0.2419±0.0094 |
| fairsteer | shared_test | 5 | 15 | 0.6026±0.1119 | 0.8306±0.1152 | 1.0000±0.0000 | 0.1194±0.1252 |
| ours_per_condition_oracle | full_test | 5 | 1328 | 0.9946±0.0054 | 0.8738±0.0109 | 0.8333±0.3333 | 0.0837±0.0194 |
| ours_predicted_condition | full_test | 5 | 1328 | 0.9946±0.0054 | 0.8732±0.0108 | 0.8333±0.3333 | 0.0843±0.0193 |
| ours_single_tau | full_test | 5 | 1328 | 0.9494±0.0126 | 0.8413±0.0184 | 0.1474±0.1322 | 0.1325±0.0240 |
| self_debiasing | shared_test | 5 | 187 | 0.9556±0.0166 | 0.1740±0.0402 | 0.4267±0.3876 | 0.8028±0.0355 |

## Reviewer-Risk Checks

- Predicted-condition classifier test accuracy: 0.9983±0.0011.
- Sub-0.05 tau_dis audit best-by-val tau: 0.0000±0.0000.
- tau_dis=0.05 test macro score in low-grid audit: 0.9342±0.0072.
- Baseline same-ID matching included 4 baseline systems; worst missing test IDs: 1318.

## Output Files

- `metrics_summary.csv`: per-seed metrics for ours and baselines.
- `aggregate_metrics.csv`: mean/std table for paper drafting.
- `paired_tests.csv`: paired bootstrap differences and p-values on matched IDs.
- `low_threshold_audit.csv`: sub-0.05 tau_dis sweep for the lower-bound concern.
- `seed_*/test_predictions.jsonl`: same-ID predictions for auditability.
