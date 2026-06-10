# Clean Defense Suite Report

This report is generated without LLM inference from saved signals/predictions.

## Same-Test-ID Aggregate Metrics

| System | Subset | Seeds | n | acc_amb | acc_dis | abs_bias_amb | FAR |
|---|---|---:|---:|---:|---:|---:|---:|
| composite | shared_test | 5 | 1328 | 0.6843±0.0138 | 0.2855±0.0109 | 0.0660±0.0280 | 0.2449±0.0164 |
| decap | shared_test | 5 | 1328 | 0.8057±0.0055 | 0.7238±0.0075 | 0.4629±0.0880 | 0.2419±0.0094 |
| fairsteer | shared_test | 5 | 1286 | 0.8513±0.0069 | 0.7185±0.0131 | 0.5532±0.0868 | 0.2591±0.0129 |
| mpt | shared_test | 5 | 196 | 0.3150±0.0619 | 0.3290±0.0292 | 0.1362±0.0512 | 0.3385±0.0594 |
| ours_per_condition_oracle | full_test | 5 | 1328 | 0.9949±0.0047 | 0.8756±0.0091 | 0.6444±0.3355 | 0.0807±0.0142 |
| ours_predicted_condition | full_test | 5 | 1328 | 0.9943±0.0046 | 0.8753±0.0095 | 0.7500±0.3191 | 0.0810±0.0147 |
| ours_single_tau | full_test | 5 | 1328 | 0.9587±0.0121 | 0.8337±0.0165 | 0.3782±0.1356 | 0.1398±0.0211 |
| self_debiasing | shared_test | 5 | 1328 | 0.9584±0.0078 | 0.1928±0.0111 | 0.2109±0.1055 | 0.7858±0.0083 |

## Reviewer-Risk Checks

- Predicted-condition classifier test accuracy: 0.9976±0.0013.
- Sub-0.05 tau_dis audit best-by-val tau: 0.0000±0.0000.
- tau_dis=0.05 test macro score in low-grid audit: 0.9352±0.0052.
- Baseline same-ID matching included 5 baseline systems; worst missing test IDs: 1147.

## Output Files

- `metrics_summary.csv`: per-seed metrics for ours and baselines.
- `aggregate_metrics.csv`: mean/std table for paper drafting.
- `paired_tests.csv`: paired bootstrap differences and p-values on matched IDs.
- `low_threshold_audit.csv`: sub-0.05 tau_dis sweep for the lower-bound concern.
- `seed_*/test_predictions.jsonl`: same-ID predictions for auditability.
