# Hybrid Abstention Audit Report

All runs use saved signals/embeddings only; no LLM inference is run.

## Clean BBQ

| System | runs | Acc_amb | Acc_dis | FAR | cond. acc. |
|---|---:|---:|---:|---:|---:|
| hybrid_uncertain_signal_fallback | 5 | 0.9979+/-0.0020 | 0.8795+/-0.0070 | 0.0723+/-0.0071 | 0.9983+/-0.0011 |
| primary_answer | 5 | 0.5596+/-0.0152 | 0.8798+/-0.0076 | 0.0717+/-0.0069 | - |
| seven_signal_moe_predicted | 5 | 0.9985+/-0.0034 | 0.8792+/-0.0076 | 0.0723+/-0.0069 | - |
| simple_condition_only | 5 | 1.0000+/-0.0000 | 0.8789+/-0.0070 | 0.0726+/-0.0067 | 0.9983+/-0.0011 |

## LOCO

| System | runs | Acc_amb | Acc_dis | FAR | cond. acc. |
|---|---:|---:|---:|---:|---:|
| hybrid_uncertain_signal_fallback | 45 | 0.9570+/-0.0441 | 0.8292+/-0.0755 | 0.1174+/-0.0552 | 0.9261+/-0.0273 |
| seven_signal_moe_predicted | 45 | 0.9463+/-0.0422 | 0.8340+/-0.0742 | 0.1127+/-0.0523 | - |
| simple_condition_only | 45 | 0.9581+/-0.0434 | 0.8268+/-0.0759 | 0.1202+/-0.0551 | 0.9261+/-0.0273 |

## Low-label Condition Classifier

| frac | system | runs | Acc_amb | Acc_dis | FAR | cond. acc. |
|---:|---|---:|---:|---:|---:|---:|
| 0.01 | hybrid_uncertain_signal_fallback | 5 | 0.9530+/-0.0054 | 0.8247+/-0.0313 | 0.1452+/-0.0339 | 0.7806+/-0.0416 |
| 0.01 | simple_condition_only | 5 | 0.9136+/-0.0051 | 0.6786+/-0.0674 | 0.2931+/-0.0680 | 0.7806+/-0.0416 |
| 0.05 | hybrid_uncertain_signal_fallback | 5 | 0.9744+/-0.0110 | 0.8548+/-0.0071 | 0.1048+/-0.0079 | 0.9355+/-0.0097 |
| 0.05 | simple_condition_only | 5 | 0.9645+/-0.0097 | 0.8301+/-0.0096 | 0.1280+/-0.0114 | 0.9355+/-0.0097 |
| 0.1 | hybrid_uncertain_signal_fallback | 5 | 0.9834+/-0.0028 | 0.8726+/-0.0053 | 0.0831+/-0.0079 | 0.9706+/-0.0075 |
| 0.1 | simple_condition_only | 5 | 0.9852+/-0.0034 | 0.8587+/-0.0133 | 0.0958+/-0.0133 | 0.9706+/-0.0075 |
| 0.25 | hybrid_uncertain_signal_fallback | 5 | 0.9952+/-0.0043 | 0.8750+/-0.0067 | 0.0786+/-0.0047 | 0.9884+/-0.0022 |
| 0.25 | simple_condition_only | 5 | 0.9913+/-0.0033 | 0.8717+/-0.0048 | 0.0807+/-0.0064 | 0.9884+/-0.0022 |
| 0.5 | hybrid_uncertain_signal_fallback | 5 | 0.9970+/-0.0024 | 0.8765+/-0.0087 | 0.0756+/-0.0070 | 0.9958+/-0.0016 |
| 0.5 | simple_condition_only | 5 | 0.9982+/-0.0013 | 0.8762+/-0.0085 | 0.0756+/-0.0073 | 0.9958+/-0.0016 |
| 1.0 | hybrid_uncertain_signal_fallback | 5 | 0.9979+/-0.0020 | 0.8795+/-0.0070 | 0.0723+/-0.0071 | 0.9983+/-0.0011 |
| 1.0 | simple_condition_only | 5 | 1.0000+/-0.0000 | 0.8789+/-0.0070 | 0.0726+/-0.0067 | 0.9983+/-0.0011 |
