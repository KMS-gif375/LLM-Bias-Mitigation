# Hybrid Abstention Audit Report

All runs use saved signals/embeddings only; no LLM inference is run.

## LOCO

| System | runs | Acc_amb | Acc_dis | FAR | cond. acc. |
|---|---:|---:|---:|---:|---:|
| hybrid_uncertain_signal_fallback | 1 | 0.9760+/-0.0000 | 0.8180+/-0.0000 | 0.1100+/-0.0000 | 0.9400+/-0.0000 |
| seven_signal_moe_predicted | 1 | 0.9760+/-0.0000 | 0.8180+/-0.0000 | 0.1100+/-0.0000 | - |
| simple_condition_only | 1 | 0.9760+/-0.0000 | 0.8180+/-0.0000 | 0.1100+/-0.0000 | 0.9400+/-0.0000 |

