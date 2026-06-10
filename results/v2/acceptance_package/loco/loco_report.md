# Clean LOCO Report

Leave-one-category-out generalization using saved signals only.

## Aggregate

| Variant | folds | acc_amb | acc_dis | FAR | condition_acc |
|---|---:|---:|---:|---:|---:|
| ours_per_condition_oracle | 45 | 0.9447+/-0.0330 | 0.8574+/-0.0747 | 0.0879+/-0.0485 | 0.9261+/-0.0273 |
| ours_predicted_condition | 45 | 0.9214+/-0.0421 | 0.8331+/-0.0793 | 0.1161+/-0.0551 | 0.9261+/-0.0273 |
| ours_single_tau | 45 | 0.8362+/-0.0536 | 0.8013+/-0.0936 | 0.1536+/-0.0777 | 0.9261+/-0.0273 |

## Files

- `loco_metrics.csv`: per-seed, per-held-out-category metrics.
- `loco_aggregate.csv`: aggregate mean/std across folds.
- `loco_residual_bias_counts.csv`: ambiguous residual bias denominator audit.
- `seed_*/folds/*/predictions.jsonl`: held-out predictions for auditability.
