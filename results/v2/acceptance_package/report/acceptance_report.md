# Acceptance Package Report

This package collects the reviewer-defense experiments and paper-ready appendix tables.

## Main Metrics

| System | subset | n | acc_amb | acc_dis | FAR | abs_bias_amb |
|---|---|---:|---:|---:|---:|---:|
| composite | shared_test | 1328.0 | 0.6843+/-0.0138 | 0.2855+/-0.0109 | 0.2449+/-0.0164 | 0.0806+/-0.0685 |
| decap | shared_test | 1328.0 | 0.8057+/-0.0055 | 0.7238+/-0.0075 | 0.2419+/-0.0094 | 0.4296+/-0.0486 |
| ours_per_condition_oracle | full_test | 1328.0 | 0.9946+/-0.0054 | 0.8738+/-0.0109 | 0.0837+/-0.0194 | 0.8333+/-0.3333 |
| ours_predicted_condition | full_test | 1328.0 | 0.9946+/-0.0054 | 0.8732+/-0.0108 | 0.0843+/-0.0193 | 0.8333+/-0.3333 |
| ours_single_tau | full_test | 1328.0 | 0.9494+/-0.0126 | 0.8413+/-0.0184 | 0.1325+/-0.0240 | 0.1474+/-0.1322 |
| self_debiasing | shared_test | 187.2 | 0.9556+/-0.0166 | 0.1740+/-0.0402 | 0.8028+/-0.0355 | 0.4267+/-0.3876 |

Auxiliary limited-overlap comparison:

| System | subset | n | acc_amb | acc_dis | FAR | note |
|---|---|---:|---:|---:|---:|---|
| fairsteer | shared_test | 14.8 | 0.6026+/-0.1119 | 0.8306+/-0.1152 | 0.1194+/-0.1252 | limited matched-ID overlap |

## Generalization

| Variant | folds | acc_amb | acc_dis | FAR |
|---|---:|---:|---:|---:|
| ours_per_condition_oracle | 45 | 0.9447+/-0.0330 | 0.8574+/-0.0747 | 0.0879+/-0.0485 |
| ours_predicted_condition | 45 | 0.9214+/-0.0421 | 0.8331+/-0.0793 | 0.1161+/-0.0551 |
| ours_single_tau | 45 | 0.8362+/-0.0536 | 0.8013+/-0.0936 | 0.1536+/-0.0777 |

## Open-BBQ Transfer

Open-BBQ n=3300: acc_amb=0.9915, acc_dis=0.8358, FAR=0.1012.

## Cross-LLM

| Model | seeds | acc_amb | acc_dis | FAR |
|---|---|---:|---:|---:|
| qwen | 42,123,456,789,999 | 0.9895+/-0.0046 | 0.8147+/-0.0183 | 0.1672+/-0.0222 |
| mistral | 42,123,456,789,999 | 0.9940+/-0.0018 | 0.7798+/-0.0099 | 0.1916+/-0.0100 |

## Generated Tables

- `main_and_baseline_metrics.csv`
- `loco_generalization_summary.csv`
- `openbbq_transfer_summary.csv`
- `cross_llm_summary.csv`
- `residual_bias_counts.csv`
- `paired_tests_summary.csv`
- `threshold_audit_summary.csv`
- `signal_ablation_summary.csv`
- `reproducibility.md`
- `claim_language.md`
